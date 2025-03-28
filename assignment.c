#include <assert.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define NUM_PROCS 4
#define CACHE_SIZE 4
#define MEM_SIZE 16
#define MSG_BUFFER_SIZE 256
#define MAX_INSTR_NUM 32

typedef unsigned char byte;

typedef enum { MODIFIED, EXCLUSIVE, SHARED, INVALID } cacheLineState;

// In addition to cache line states, each directory entry also has a state
// EM ( exclusive or modified ) : memory block is in only one cache.
//                  When a memory block in a cache is written to, it would have
//                  resulted in cache hit and transition from EXCLUSIVE to MODIFIED.
//                  Main memory / directory has no way of knowing of this transition.
//                  Hence, this state only cares that the memory block is in a single
//                  cache, and not whether its in EXCLUSIVE or MODIFIED.
// S ( shared )     : memory block is in multiple caches
// U ( unowned )    : memory block is not in any cache
typedef enum { EM, S, U } directoryEntryState;

typedef enum { 
    READ_REQUEST,       // requesting node sends to home node on a read miss 
    WRITE_REQUEST,      // requesting node sends to home node on a write miss 
    REPLY_RD,           // home node replies with data to requestor for read request
    REPLY_WR,           // home node replies to requestor for write request
    REPLY_ID,           // home node replies with IDs of sharers to requestor
    INV,                // owner node asks sharers to invalidate
    UPGRADE,            // owner node asks home node to change state to EM
    WRITEBACK_INV,      // home node asks owner node to flush and change to INVALID
    WRITEBACK_INT,      // home node asks owner node to flush and change to SHARED
    FLUSH,              // owner flushes data to home + requestor
    FLUSH_INVACK,       // flush, piggybacking an InvAck message
    EVICT_SHARED,       // handle cache replacement of a shared cache line
    EVICT_MODIFIED      // handle cache replacement of a modified cache line
} transactionType;

// We will create our own address space which will be of size 1 byte
// LSB 4 bits indicate the location in memory
// MSB 4 bits indicate the processor it is present in.
// For example, 0x36 means the memory block at index 6 in the 3rd processor
typedef struct instruction {
    byte type;      // 'R' for read, 'W' for write
    byte address;
    byte value;     // used only for write operations
} instruction;

typedef struct cacheLine {
    byte address;           // this is the address in memory
    byte value;             // this is the value stored in cached memory
    cacheLineState state;   // state for you to implement MESI protocol
} cacheLine;

typedef struct directoryEntry {
    byte bitVector;         // each bit indicates whether that processor has this
                            // memory block in its cache
    directoryEntryState state;
} directoryEntry;

// Note that each message will contain values only in the fields which are relevant 
// to the transactionType
typedef struct message {
    transactionType type;
    int sender;          // thread id that sent the message
    byte address;        // memory block address
    byte value;          // value in memory / cache
    byte bitVector;      // ids of sharer nodes
    int secondReceiver;  // used when you need to send a message to 2 nodes, where
                         // 1 node id is in the sender field
    directoryEntryState dirState;   // directory entry state of the memory block
} message;

typedef struct messageBuffer {
    message queue[ MSG_BUFFER_SIZE ];
    // a circular queue message buffer
    int head;
    int tail;
    int count;          // store total number of messages processed by the node
} messageBuffer;

typedef struct processorNode {
    cacheLine cache[ CACHE_SIZE ];
    byte memory[ MEM_SIZE ];
    directoryEntry directory[ MEM_SIZE ];
    instruction instructions[ MAX_INSTR_NUM ];
    int instructionCount;
} processorNode;

void initializeProcessor( int threadId, processorNode *node, char *dirName );
void sendMessage( int receiver, message msg );  // IMPLEMENT
void handleCacheReplacement( int sender, cacheLine oldCacheLine );  // IMPLEMENT
void printProcessorState( int processorId, processorNode node );

messageBuffer messageBuffers[ NUM_PROCS ];
// Create locks to ensure thread-safe access to each processor's message buffer.
omp_lock_t msgBufferLocks[ NUM_PROCS ];

int main( int argc, char * argv[] ) {
    if (argc < 2) {
        fprintf( stderr, "Usage: %s <test_directory>\n", argv[0] );
        return EXIT_FAILURE;
    }
    char *dirName = argv[1];
    
    // Set number of threads to NUM_PROCS
    omp_set_num_threads(NUM_PROCS);

    for ( int i = 0; i < NUM_PROCS; i++ ) {
        messageBuffers[ i ].count = 0;
        messageBuffers[ i ].head = 0;
        messageBuffers[ i ].tail = 0;
        // Initialize the locks in msgBufferLocks
        omp_init_lock(&msgBufferLocks[i]);
    }
    processorNode node;

    // Create the omp parallel region
    #pragma omp parallel private(node)
    {
        int threadId = omp_get_thread_num();
        initializeProcessor( threadId, &node, dirName );
        
        // Wait for all processors to complete initialization
        #pragma omp barrier

        message msg;
        message msgReply;
        instruction instr;
        int instructionIdx = -1;
        int printProcState = 1;
        byte waitingForReply = 0;
        while ( 1 ) {
            // Process all messages in message queue first
            while ( 
                messageBuffers[ threadId ].count > 0 &&
                messageBuffers[ threadId ].head != messageBuffers[ threadId ].tail
            ) {
                int head = messageBuffers[ threadId ].head;
                msg = messageBuffers[ threadId ].queue[ head ];
                messageBuffers[ threadId ].head = ( head + 1 ) % MSG_BUFFER_SIZE;

                #ifdef DEBUG
                printf( "Processor %d msg from: %d, type: %d, address: 0x%02X\n",
                        threadId, msg.sender, msg.type, msg.address );
                #endif /* ifdef DEBUG */

                // extract procNodeAddr and memBlockAddr from message address
                byte procNodeAddr = (msg.address >> 4) & 0x0F;
                byte memBlockAddr = msg.address & 0x0F;
                byte cacheIndex = memBlockAddr % CACHE_SIZE;

                switch ( msg.type ) {
                    case READ_REQUEST:
                        // Handle read request in home node
                        switch (node.directory[memBlockAddr].state) {
                            case U:
                            case S:
                                // Update directory state and bitvector
                                if (node.directory[memBlockAddr].state == U) {
                                    node.directory[memBlockAddr].state = S;
                                }
                                node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                                
                                // Send reply with data
                                msgReply.type = REPLY_RD;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.value = node.memory[memBlockAddr];
                                sendMessage(msg.sender, msgReply);
                                break;
                                
                            case EM:
                                // Forward request to current owner
                                msgReply.type = WRITEBACK_INT;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.secondReceiver = msg.sender;
                                
                                // Find the owner processor
                                byte owner;
                                for (owner = 0; owner < NUM_PROCS; owner++) {
                                    if (node.directory[memBlockAddr].bitVector & (1 << owner)) {
                                        break;
                                    }
                                }
                                sendMessage(owner, msgReply);
                                break;
                        }
                        break;

                    case WRITE_REQUEST:
                        // Handle write request in home node
                        switch (node.directory[memBlockAddr].state) {
                            case U:
                                // No cache has this block, make requestor the owner
                                node.directory[memBlockAddr].state = EM;
                                node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                                node.memory[memBlockAddr] = msg.value;  // Update memory
                                
                                // Send reply
                                msgReply.type = REPLY_WR;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.value = msg.value;
                                sendMessage(msg.sender, msgReply);
                                break;
                                
                            case S:
                                // Send sharers list to new owner
                                msgReply.type = REPLY_ID;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.bitVector = node.directory[memBlockAddr].bitVector & ~(1 << msg.sender);
                                msgReply.value = msg.value;
                                
                                // Update directory and memory
                                node.directory[memBlockAddr].state = EM;
                                node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                                node.memory[memBlockAddr] = msg.value;
                                
                                sendMessage(msg.sender, msgReply);
                                break;
                                
                            case EM:
                                // Ask current owner to writeback and invalidate
                                msgReply.type = WRITEBACK_INV;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.secondReceiver = msg.sender;
                                
                                // Find the owner processor
                                byte owner;
                                for (owner = 0; owner < NUM_PROCS; owner++) {
                                    if (node.directory[memBlockAddr].bitVector & (1 << owner)) {
                                        break;
                                    }
                                }
                                sendMessage(owner, msgReply);
                                break;
                        }
                        break;

                    case REPLY_RD:
                        // Handle cache replacement if needed
                        if (node.cache[cacheIndex].state != INVALID) {
                            handleCacheReplacement(threadId, node.cache[cacheIndex]);
                        }
                        
                        // Update cache with new data
                        node.cache[cacheIndex].address = msg.address;
                        node.cache[cacheIndex].value = msg.value;
                        node.cache[cacheIndex].state = SHARED;
                        waitingForReply = 0;
                        break;

                    case WRITEBACK_INT:
                        // Send FLUSH to both home and requesting nodes
                        msgReply.type = FLUSH;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.value = node.cache[cacheIndex].value;
                        
                        // Send to home node
                        sendMessage(procNodeAddr, msgReply);
                        
                        // Send to requesting node if different from home
                        if (msg.secondReceiver != procNodeAddr) {
                            sendMessage(msg.secondReceiver, msgReply);
                        }
                        
                        // Update cache state to SHARED
                        node.cache[cacheIndex].state = SHARED;
                        break;

                    case FLUSH:
                        if (threadId == procNodeAddr) {
                            // In home node, update memory and directory
                            node.memory[memBlockAddr] = msg.value;
                            node.directory[memBlockAddr].state = S;
                            node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                            if (msg.secondReceiver != procNodeAddr) {
                                node.directory[memBlockAddr].bitVector |= (1 << msg.secondReceiver);
                            }
                        } else {
                            // In requesting node, update cache
                            if (node.cache[cacheIndex].state != INVALID) {
                                handleCacheReplacement(threadId, node.cache[cacheIndex]);
                            }
                            node.cache[cacheIndex].address = msg.address;
                            node.cache[cacheIndex].value = msg.value;
                            node.cache[cacheIndex].state = SHARED;
                            waitingForReply = 0;
                        }
                        break;

                    case UPGRADE:
                        // Update directory state and send sharers list
                        msgReply.type = REPLY_ID;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.bitVector = node.directory[memBlockAddr].bitVector & ~(1 << msg.sender);
                        
                        // Update directory
                        node.directory[memBlockAddr].state = EM;
                        node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                        
                        sendMessage(msg.sender, msgReply);
                        break;

                    case REPLY_ID:
                        // Send invalidation to all sharers
                        msgReply.type = INV;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        
                        // Send INV to each sharer
                        for (int i = 0; i < NUM_PROCS; i++) {
                            if (msg.bitVector & (1 << i)) {
                                sendMessage(i, msgReply);
                            }
                        }
                        
                        // Update cache state to MODIFIED
                        node.cache[cacheIndex].state = MODIFIED;
                        waitingForReply = 0;
                        break;

                    case INV:
                        // Invalidate cache entry if it exists
                        if (node.cache[cacheIndex].address == msg.address) {
                            node.cache[cacheIndex].state = INVALID;
                        }
                        break;

                    case REPLY_WR:
                        // Handle cache replacement if needed
                        if (node.cache[cacheIndex].state != INVALID) {
                            handleCacheReplacement(threadId, node.cache[cacheIndex]);
                        }
                        
                        // Update cache with new data
                        node.cache[cacheIndex].address = msg.address;
                        node.cache[cacheIndex].value = msg.value;
                        node.cache[cacheIndex].state = MODIFIED;
                        
                        // Update memory if this is the home node
                        if (threadId == procNodeAddr) {
                            node.memory[memBlockAddr] = msg.value;
                        }
                        waitingForReply = 0;
                        break;

                    case WRITEBACK_INV:
                        // Send FLUSH to both home and requesting nodes
                        msgReply.type = FLUSH;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.value = node.cache[cacheIndex].value;
                        
                        // Send to home node
                        sendMessage(procNodeAddr, msgReply);
                        
                        // Send to requesting node if different from home
                        if (msg.secondReceiver != procNodeAddr) {
                            sendMessage(msg.secondReceiver, msgReply);
                        }
                        
                        // Invalidate cache entry
                        node.cache[cacheIndex].state = INVALID;
                        break;

                    case FLUSH_INVACK:
                        if (threadId == procNodeAddr) {
                            // In home node, update directory and memory
                            node.memory[memBlockAddr] = msg.value;
                            node.directory[memBlockAddr].state = EM;
                            node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                        } else {
                            // In requesting node, update cache
                            if (node.cache[cacheIndex].state != INVALID) {
                                handleCacheReplacement(threadId, node.cache[cacheIndex]);
                            }
                            node.cache[cacheIndex].address = msg.address;
                            node.cache[cacheIndex].value = msg.value;
                            node.cache[cacheIndex].state = MODIFIED;
                            waitingForReply = 0;
                        }
                        break;
                    
                    case EVICT_SHARED:
                        if (threadId == procNodeAddr) {
                            // In home node
                            node.directory[memBlockAddr].bitVector &= ~(1 << msg.sender);
                            
                            // Count remaining sharers
                            int sharerCount = 0;
                            for (int i = 0; i < NUM_PROCS; i++) {
                                if (node.directory[memBlockAddr].bitVector & (1 << i)) {
                                    sharerCount++;
                                }
                            }
                            
                            if (sharerCount == 0) {
                                node.directory[memBlockAddr].state = U;
                            } else if (sharerCount == 1) {
                                node.directory[memBlockAddr].state = EM;
                                // Find the remaining sharer
                                for (int i = 0; i < NUM_PROCS; i++) {
                                    if (node.directory[memBlockAddr].bitVector & (1 << i)) {
                                        msgReply.type = EVICT_SHARED;
                                        msgReply.sender = threadId;
                                        msgReply.address = msg.address;
                                        sendMessage(i, msgReply);
                                        break;
                                    }
                                }
                            }
                        } else {
                            // In remaining sharer (new owner)
                            node.cache[cacheIndex].state = EXCLUSIVE;
                        }
                        break;

                    case EVICT_MODIFIED:
                        // This is in home node
                        node.memory[memBlockAddr] = msg.value;
                        node.directory[memBlockAddr].state = U;
                        node.directory[memBlockAddr].bitVector = 0;
                        break;
                }
            }
            
            // Check if we are waiting for a reply message
            // if yes, then we have to complete the previous instruction before
            // moving on to the next
            if ( waitingForReply > 0 ) {
                continue;
            }

            // Process next instruction if not waiting for reply
            if (!waitingForReply && instructionIdx < node.instructionCount - 1) {
                instructionIdx++;
                instr = node.instructions[instructionIdx];
                
                // Extract processor and memory block address
                byte procNodeAddr = (instr.address >> 4) & 0x0F;
                byte memBlockAddr = instr.address & 0x0F;
                byte cacheIndex = memBlockAddr % CACHE_SIZE;
                
                // Handle instruction
                switch (instr.type) {
                    case 'R':
                        // Handle read instruction
                        if (node.cache[cacheIndex].state == INVALID ||
                            node.cache[cacheIndex].address != instr.address) {
                            // Cache miss, send read request
                            msg.type = READ_REQUEST;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        }
                        break;
                        
                    case 'W':
                        // Handle write instruction
                        if (node.cache[cacheIndex].state == INVALID ||
                            node.cache[cacheIndex].address != instr.address) {
                            // Cache miss, send write request
                            msg.type = WRITE_REQUEST;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            msg.value = instr.value;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        } else if (node.cache[cacheIndex].state == SHARED) {
                            // Cache hit but need to upgrade
                            msg.type = UPGRADE;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        } else {
                            // Cache hit in EXCLUSIVE or MODIFIED state
                            node.cache[cacheIndex].value = instr.value;
                            node.cache[cacheIndex].state = MODIFIED;
                        }
                        break;
                }
            }

            // Check if we are waiting for a reply message
            // if yes, then we have to complete the previous instruction before
            // moving on to the next
            if ( waitingForReply > 0 ) {
                continue;
            }

            // Process an instruction
            if ( instructionIdx < node.instructionCount - 1 ) {
                instructionIdx++;
            } else {
                if ( printProcState > 0 ) {
                    printProcessorState( threadId, node );
                    printProcState--;
                }
                // even though there are no more instructions, this processor might
                // still need to react to new transaction messages
                //
                // once all the processors are done printing and appear to have no
                // more network transactions, please terminate the program by sending
                // a SIGINT ( CTRL+C )
                continue;
            }
        }
    }
}

void sendMessage( int receiver, message msg ) {
    // Get the message buffer for the receiver
    messageBuffer *buffer = &messageBuffers[receiver];
    
    // Check if buffer is full
    if ((buffer->tail + 1) % MSG_BUFFER_SIZE == buffer->head) {
        fprintf(stderr, "Error: Message buffer full for processor %d\n", receiver);
        return;
    }
    
    // Add message to buffer
    buffer->queue[buffer->tail] = msg;
    buffer->tail = (buffer->tail + 1) % MSG_BUFFER_SIZE;
    buffer->count++;
}

void handleCacheReplacement( int sender, cacheLine oldCacheLine ) {
    // If cache line is invalid, nothing to do
    if (oldCacheLine.state == INVALID) {
        return;
    }

    // Extract processor and memory block address
    byte procNodeAddr = (oldCacheLine.address >> 4) & 0x0F;
    byte memBlockAddr = oldCacheLine.address & 0x0F;
    
    message msg;
    msg.sender = sender;
    msg.address = oldCacheLine.address;
    msg.value = oldCacheLine.value;
    
    // Handle based on cache line state
    switch (oldCacheLine.state) {
        case MODIFIED:
            msg.type = EVICT_MODIFIED;
            sendMessage(procNodeAddr, msg);
            break;
            
        case EXCLUSIVE:
        case SHARED:
            msg.type = EVICT_SHARED;
            sendMessage(procNodeAddr, msg);
            break;
            
        default:
            break;
    }
}

void initializeProcessor( int threadId, processorNode *node, char *dirName ) {
    // Initialize memory
    for ( int i = 0; i < MEM_SIZE; i++ ) {
        node->memory[i] = i + threadId * 20;  // Initialize with unique values
    }

    // Initialize directory
    for ( int i = 0; i < MEM_SIZE; i++ ) {
        node->directory[i].state = U;
        node->directory[i].bitVector = 0;
    }

    // Initialize cache
    for ( int i = 0; i < CACHE_SIZE; i++ ) {
        node->cache[i].state = INVALID;
        node->cache[i].address = 0xFF;  // Invalid address
        node->cache[i].value = 0;
    }

    // Read instructions
    char fileName[256];
    sprintf( fileName, "%s/core_%d.txt", dirName, threadId );
    FILE *fp = fopen( fileName, "r" );
    if ( fp == NULL ) {
        fprintf( stderr, "Error opening file %s\n", fileName );
        exit( EXIT_FAILURE );
    }

    char line[256];
    int count = 0;
    while ( fgets( line, sizeof( line ), fp ) ) {
        if ( line[0] == 'R' ) {
            node->instructions[count].type = 'R';
            sscanf( line + 3, "%hhx", &node->instructions[count].address );
        } else if ( line[0] == 'W' ) {
            node->instructions[count].type = 'W';
            sscanf( line + 3, "%hhx %hhu",
                    &node->instructions[count].address,
                    &node->instructions[count].value );
        }
        count++;
    }
    node->instructionCount = count;
    fclose( fp );
}

void printProcessorState( int processorId, processorNode node ) {
    const char *dirStateStr[] = { "EM", "S", "U" };
    const char *cacheStateStr[] = { "MODIFIED", "EXCLUSIVE", "SHARED", "INVALID" };

    printf( "=======================================\n" );
    printf( " Processor Node: %d\n", processorId );
    printf( "=======================================\n\n" );

    // Print memory state
    printf( "-------- Memory State -------\n" );
    printf( "| Index | Address | Value   |\n" );
    printf( "|---------------------------|\n" );
    for ( int i = 0; i < MEM_SIZE; i++ ) {
        printf( "|  %3d  |  0x%02X   |  %5d  |\n", i,
                ( processorId << 4 ) + i, node.memory[i] );
    }
    printf( "-----------------------------\n\n" );

    // Print directory state
    printf( "------------ Directory State --------------\n" );
    printf( "| Index | Address | State | BitVector     |\n" );
    printf( "|-----------------------------------------|\n" );
    for ( int i = 0; i < MEM_SIZE; i++ ) {
        printf( "|  %3d  |  0x%02X   |  %2s  |   0x%08X   |\n", i,
                ( processorId << 4 ) + i, dirStateStr[node.directory[i].state],
                node.directory[i].bitVector );
    }
    printf( "-------------------------------------------\n\n" );

    // Print cache state
    printf( "------------ Cache State ----------------\n" );
    printf( "| Index | Address | Value |  State      |\n" );
    printf( "|---------------------------------------|\n" );
    for ( int i = 0; i < CACHE_SIZE; i++ ) {
        printf( "|  %3d  |  0x%02X   |  %3d  |  %-9s  |\n", i,
                node.cache[i].address, node.cache[i].value,
                cacheStateStr[node.cache[i].state] );
    }
    printf( "----------------------------------------\n\n" );
}

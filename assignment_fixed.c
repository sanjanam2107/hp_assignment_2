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

typedef struct message {
    transactionType type;
    int sender;          // thread id that sent the message
    byte address;        // memory block address
    byte value;          // value in memory / cache
    byte bitVector;      // ids of sharer nodes
    int secondReceiver;  // used when you need to send a message to 2 nodes
    directoryEntryState dirState;   // directory entry state of the memory block
} message;

typedef struct messageBuffer {
    message queue[MSG_BUFFER_SIZE];
    int head;
    int tail;
    int count;          // store total number of messages processed by the node
} messageBuffer;

typedef struct processorNode {
    cacheLine cache[CACHE_SIZE];
    byte memory[MEM_SIZE];
    directoryEntry directory[MEM_SIZE];
    instruction instructions[MAX_INSTR_NUM];
    int instructionCount;
} processorNode;

void initializeProcessor(int threadId, processorNode *node, char *dirName);
void sendMessage(int receiver, message msg);
void handleCacheReplacement(int sender, cacheLine oldCacheLine);
void printProcessorState(int processorId, processorNode node);

messageBuffer messageBuffers[NUM_PROCS];
omp_lock_t msgBufferLocks[NUM_PROCS];

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_directory>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *dirName = argv[1];
    
    omp_set_num_threads(NUM_PROCS);

    for (int i = 0; i < NUM_PROCS; i++) {
        messageBuffers[i].count = 0;
        messageBuffers[i].head = 0;
        messageBuffers[i].tail = 0;
        omp_init_lock(&msgBufferLocks[i]);
    }
    processorNode node;

    #pragma omp parallel private(node)
    {
        int threadId = omp_get_thread_num();
        initializeProcessor(threadId, &node, dirName);
        
        #pragma omp barrier

        message msg;
        message msgReply;
        instruction instr;
        int instructionIdx = -1;
        int printProcState = 1;
        byte waitingForReply = 0;
        while (1) {
            while (messageBuffers[threadId].count > 0 &&
                   messageBuffers[threadId].head != messageBuffers[threadId].tail) {
                int head = messageBuffers[threadId].head;
                msg = messageBuffers[threadId].queue[head];
                messageBuffers[threadId].head = (head + 1) % MSG_BUFFER_SIZE;

                byte procNodeAddr = (msg.address >> 4) & 0x0F;
                byte memBlockAddr = msg.address & 0x0F;
                byte cacheIndex = memBlockAddr % CACHE_SIZE;

                // Check if this is a valid memory access for this processor
                if (procNodeAddr != threadId && msg.type != READ_REQUEST && msg.type != WRITE_REQUEST) {
                    continue;
                }

                switch (msg.type) {
                    case READ_REQUEST:
                        switch (node.directory[memBlockAddr].state) {
                            case U:
                                // No cache has this block, make requestor the owner
                                node.directory[memBlockAddr].state = EM;
                                node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                                node.memory[memBlockAddr] = msg.value;
                                
                                msgReply.type = REPLY_RD;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.value = node.memory[memBlockAddr];
                                sendMessage(msg.sender, msgReply);
                                break;
                                
                            case S:
                                // Add requestor to sharers
                                node.directory[memBlockAddr].bitVector |= (1 << msg.sender);
                                
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
                                
                                int owner;
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
                        switch (node.directory[memBlockAddr].state) {
                            case U:
                                // No cache has this block, make requestor the owner
                                node.directory[memBlockAddr].state = EM;
                                node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                                node.memory[memBlockAddr] = msg.value;
                                
                                msgReply.type = REPLY_WR;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.value = msg.value;
                                sendMessage(msg.sender, msgReply);
                                break;
                                
                            case S:
                                // Invalidate all sharers and make requestor the owner
                                msgReply.type = REPLY_ID;
                                msgReply.sender = threadId;
                                msgReply.address = msg.address;
                                msgReply.bitVector = node.directory[memBlockAddr].bitVector & ~(1 << msg.sender);
                                msgReply.value = msg.value;
                                
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
                        if (node.cache[cacheIndex].state != INVALID) {
                            handleCacheReplacement(threadId, node.cache[cacheIndex]);
                        }
                        
                        node.cache[cacheIndex].address = msg.address;
                        node.cache[cacheIndex].value = msg.value;
                        node.cache[cacheIndex].state = SHARED;
                        waitingForReply = 0;
                        break;

                    case WRITEBACK_INT:
                        msgReply.type = FLUSH;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.value = node.cache[cacheIndex].value;
                        
                        sendMessage(procNodeAddr, msgReply);
                        
                        if (msg.secondReceiver != procNodeAddr) {
                            sendMessage(msg.secondReceiver, msgReply);
                        }
                        
                        node.cache[cacheIndex].state = SHARED;
                        break;

                    case FLUSH:
                        if (threadId == procNodeAddr) {
                            node.memory[memBlockAddr] = msg.value;
                            node.directory[memBlockAddr].state = S;
                            node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                            if (msg.secondReceiver != procNodeAddr) {
                                node.directory[memBlockAddr].bitVector |= (1 << msg.secondReceiver);
                            }
                        } else {
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
                        msgReply.type = REPLY_ID;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.bitVector = node.directory[memBlockAddr].bitVector & ~(1 << msg.sender);
                        
                        node.directory[memBlockAddr].state = EM;
                        node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                        
                        sendMessage(msg.sender, msgReply);
                        break;

                    case REPLY_ID:
                        msgReply.type = INV;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        
                        for (int i = 0; i < NUM_PROCS; i++) {
                            if (msg.bitVector & (1 << i)) {
                                sendMessage(i, msgReply);
                            }
                        }
                        
                        node.cache[cacheIndex].state = MODIFIED;
                        waitingForReply = 0;
                        break;

                    case INV:
                        if (node.cache[cacheIndex].address == msg.address) {
                            node.cache[cacheIndex].state = INVALID;
                        }
                        break;

                    case REPLY_WR:
                        if (node.cache[cacheIndex].state != INVALID) {
                            handleCacheReplacement(threadId, node.cache[cacheIndex]);
                        }
                        
                        node.cache[cacheIndex].address = msg.address;
                        node.cache[cacheIndex].value = msg.value;
                        node.cache[cacheIndex].state = MODIFIED;
                        
                        if (threadId == procNodeAddr) {
                            node.memory[memBlockAddr] = msg.value;
                        }
                        waitingForReply = 0;
                        break;

                    case WRITEBACK_INV:
                        msgReply.type = FLUSH;
                        msgReply.sender = threadId;
                        msgReply.address = msg.address;
                        msgReply.value = node.cache[cacheIndex].value;
                        
                        sendMessage(procNodeAddr, msgReply);
                        
                        if (msg.secondReceiver != procNodeAddr) {
                            sendMessage(msg.secondReceiver, msgReply);
                        }
                        
                        node.cache[cacheIndex].state = INVALID;
                        break;

                    case FLUSH_INVACK:
                        if (threadId == procNodeAddr) {
                            node.memory[memBlockAddr] = msg.value;
                            node.directory[memBlockAddr].state = EM;
                            node.directory[memBlockAddr].bitVector = (1 << msg.sender);
                        } else {
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
                            node.directory[memBlockAddr].bitVector &= ~(1 << msg.sender);
                            
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
                            node.cache[cacheIndex].state = EXCLUSIVE;
                        }
                        break;

                    case EVICT_MODIFIED:
                        if (threadId == procNodeAddr) {
                            node.memory[memBlockAddr] = msg.value;
                            node.directory[memBlockAddr].state = U;
                            node.directory[memBlockAddr].bitVector = 0;
                        }
                        break;
                }
            }
            
            if (waitingForReply > 0) {
                continue;
            }

            if (!waitingForReply && instructionIdx < node.instructionCount - 1) {
                instructionIdx++;
                instr = node.instructions[instructionIdx];
                
                byte procNodeAddr = (instr.address >> 4) & 0x0F;
                byte memBlockAddr = instr.address & 0x0F;
                byte cacheIndex = memBlockAddr % CACHE_SIZE;
                
                // Check if this is a valid memory access for this processor
                if (procNodeAddr != threadId) {
                    continue;
                }
                
                switch (instr.type) {
                    case 'R':
                        if (node.cache[cacheIndex].state == INVALID ||
                            node.cache[cacheIndex].address != instr.address) {
                            msg.type = READ_REQUEST;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        }
                        break;
                        
                    case 'W':
                        if (node.cache[cacheIndex].state == INVALID ||
                            node.cache[cacheIndex].address != instr.address) {
                            msg.type = WRITE_REQUEST;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            msg.value = instr.value;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        } else if (node.cache[cacheIndex].state == SHARED) {
                            msg.type = UPGRADE;
                            msg.sender = threadId;
                            msg.address = instr.address;
                            sendMessage(procNodeAddr, msg);
                            waitingForReply = 1;
                        } else {
                            node.cache[cacheIndex].value = instr.value;
                            node.cache[cacheIndex].state = MODIFIED;
                        }
                        break;
                }
            }

            if (waitingForReply > 0) {
                continue;
            }

            if (instructionIdx < node.instructionCount - 1) {
                instructionIdx++;
            } else {
                if (printProcState > 0) {
                    printProcessorState(threadId, node);
                    printProcState--;
                }
                continue;
            }
        }
    }
}

void sendMessage(int receiver, message msg) {
    messageBuffer *buffer = &messageBuffers[receiver];
    
    if ((buffer->tail + 1) % MSG_BUFFER_SIZE == buffer->head) {
        fprintf(stderr, "Error: Message buffer full for processor %d\n", receiver);
        return;
    }
    
    buffer->queue[buffer->tail] = msg;
    buffer->tail = (buffer->tail + 1) % MSG_BUFFER_SIZE;
    buffer->count++;
}

void handleCacheReplacement(int sender, cacheLine oldCacheLine) {
    if (oldCacheLine.state == INVALID) {
        return;
    }

    byte procNodeAddr = (oldCacheLine.address >> 4) & 0x0F;
    byte memBlockAddr = oldCacheLine.address & 0x0F;
    
    message msg;
    msg.sender = sender;
    msg.address = oldCacheLine.address;
    msg.value = oldCacheLine.value;
    
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

void initializeProcessor(int threadId, processorNode *node, char *dirName) {
    // Initialize memory with unique values for each processor
    for (int i = 0; i < MEM_SIZE; i++) {
        node->memory[i] = i + threadId * 20;
    }

    // Initialize directory
    for (int i = 0; i < MEM_SIZE; i++) {
        node->directory[i].state = U;
        node->directory[i].bitVector = 0;
    }

    // Initialize cache
    for (int i = 0; i < CACHE_SIZE; i++) {
        node->cache[i].state = INVALID;
        node->cache[i].address = 0xFF;
        node->cache[i].value = 0;
    }

    // Read instructions
    char fileName[256];
    sprintf(fileName, "%s/core_%d.txt", dirName, threadId);
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'R') {
            node->instructions[count].type = 'R';
            sscanf(line + 3, "%hhx", &node->instructions[count].address);
        } else if (line[0] == 'W') {
            node->instructions[count].type = 'W';
            sscanf(line + 3, "%hhx %hhu",
                   &node->instructions[count].address,
                   &node->instructions[count].value);
        }
        count++;
    }
    node->instructionCount = count;
    fclose(fp);
}

void printProcessorState(int processorId, processorNode node) {
    static const char *cacheStateStr[] = { "MODIFIED", "EXCLUSIVE", "SHARED", "INVALID" };
    static const char *dirStateStr[] = { "EM", "S", "U" };

    #pragma omp critical
    {
    printf("=======================================\n");
    printf(" Processor Node: %d\n", processorId);
    printf("=======================================\n\n");

    // Print memory state
    printf("-------- Memory State --------\n");
    printf("| Index | Address |   Value  |\n");
    printf("|----------------------------|\n");
    for (int i = 0; i < MEM_SIZE; i++) {
        printf("|  %3d  |  0x%02X   |  %5d   |\n", i, (processorId << 4) + i,
                node.memory[i]);
    }
    printf("------------------------------\n\n");

    // Print directory state
    printf("------------ Directory State ---------------\n");
    printf("| Index | Address | State |    BitVector   |\n");
    printf("|------------------------------------------|\n");
    for (int i = 0; i < MEM_SIZE; i++) {
        printf("|  %3d  |  0x%02X   |  %2s   |   0x%08B   |\n", i,
                (processorId << 4) + i, dirStateStr[node.directory[i].state],
                node.directory[i].bitVector);
    }
    printf("--------------------------------------------\n\n");
    
    // Print cache state
    printf("------------ Cache State ----------------\n");
    printf("| Index | Address | Value |    State    |\n");
    printf("|---------------------------------------|\n");
    for (int i = 0; i < CACHE_SIZE; i++) {
        printf("|  %3d  |  0x%02X   |  %3d  |  %8s \t|\n", 
               i, node.cache[i].address, node.cache[i].value,
               cacheStateStr[node.cache[i].state]);
    }
    printf("----------------------------------------\n\n");
    }
} 
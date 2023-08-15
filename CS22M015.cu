/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
#define INTMAX 1000000007

using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
/* In this kernel we are calculating last node of each level using the 
assumption that nodes are sequential on each level.*/
__global__ void findLastNodeOfEachLevel(int *d_lastNode,int firstNode, int lastNode, int *d_offset, int *csr){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    node+=firstNode;
    if(node<=lastNode){
        /* Using offset and csr to compute the edges of each node in current level*/
        for(int i=d_offset[node]; i<d_offset[node+1];i++){
            atomicMax(&d_lastNode[0], csr[i]);
        }
    } 
}
       

/* Here we are activating level 0 nodes and calculating indegree of level 1. */
__global__ void activateLevel0(int firstNode, int lastNode, int V, int currLevel, int *aid,
 bool *activeness, int *csr, int *offset){

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    node+=firstNode;
    if(node<=lastNode){
        aid[node]=0;
        activeness[node]=true;
        /* Using offset and csr to compute the edges of each node in current level*/
        for(int i=offset[node]; i<offset[node+1];i++){
            atomicInc((unsigned int *)&aid[csr[i]],INTMAX); //aid[csr[i]]++
        }
    }
}

/* Here we are activating level 1 - (L-1) nodes and calculating indegree of 
next level. */
__global__ void activateRestOfTheLevels(int firstNode, int lastNode, int i, int *offset, int *csr
, int *aid, int *apr, bool *activeness){

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    node+=firstNode;
    if(node<=lastNode){
        // Applying rule 1. 
        if(aid[node]>= apr[node]){
            activeness[node]=true;   
        }
    }
}

/* Here we are applying second rule.*/
__global__ void deactivateRestOfTheLevels(int firstNode, int lastNode, int i, int *offset, int *csr,
int *aid, int *apr, bool *activeness){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    node+=firstNode;
    if(node<=lastNode){
        // Here we are deactivating the nodes.
        if((node-1)>=firstNode && (node+1)<=lastNode && activeness[node-1]==false
            && activeness[node+1]==false){
            activeness[node]=false;
        }

        /* If node is still active then in that case increase active indegree
        of nodes of next level*/
        if(activeness[node]==true){
            for(int i=offset[node]; i<offset[node+1];i++){
                atomicInc((unsigned int *)&aid[csr[i]],INTMAX);//atomic_inc(); //aid[csr[i]]++
            }
        }
    }
}
    

/* In this kernel we are counting the active nodes on each level after 
doing all the processing. */
__global__ void countActiveNodes(int currLevel,int firstNode, int lastNode, 
  int *d_activeVertex, bool *activeness){

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    node+=firstNode;

    if(node<=lastNode){
      if(activeness[node]==true){
        atomicInc((unsigned int *)&d_activeVertex[currLevel],INTMAX);
      }
    }

  }
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

/* Init. d_activeVertex with 0*/
cudaMemset(d_activeVertex, 0, (L)*sizeof(int));
bool *d_activeness;
/* activeness array tells us which nodes are active/unactive*/
cudaMalloc(&d_activeness, (V)*sizeof(bool));
cudaMemset(d_activeness, 0, (V)*sizeof(bool));

/* Find last node of level 0*/
int lastNodeOfLevel0=0;
while(h_apr[lastNodeOfLevel0]==0){
    lastNodeOfLevel0++;
}


/* Here we have declared all the variables that 
are required to create and build the array to store last node of each level.*/

int *h_lastNodeOfEachLevel; // stores last node of each level.

h_lastNodeOfEachLevel = (int *)malloc((L+1)*sizeof(int)); // L size
h_lastNodeOfEachLevel[L] = -1;

h_lastNodeOfEachLevel[0]=lastNodeOfLevel0-1;
int firstNode = 0;

int *h_lastNode, *d_lastNode;
cudaMalloc(&d_lastNode, sizeof(int));
h_lastNode = (int *)malloc(sizeof(int));
h_lastNode[0] = lastNodeOfLevel0 -1;
cudaMemcpy(d_lastNode, h_lastNode, sizeof(int), cudaMemcpyHostToDevice);
int lastNode = lastNodeOfLevel0-1;

for(int i=0;i<L-1;i++){
    int totalThreads = h_lastNodeOfEachLevel[i] - firstNode + 1;
    int totalBlocks = (totalThreads + 1024-1)/1024;
    findLastNodeOfEachLevel<<<totalBlocks, 1024>>>(d_lastNode,firstNode, lastNode, d_offset, d_csrList);
    cudaDeviceSynchronize();
    firstNode = lastNode + 1;
    cudaMemcpy(h_lastNode, d_lastNode, sizeof(int), cudaMemcpyDeviceToHost);
    h_lastNodeOfEachLevel[i+1] = h_lastNode[0];
    lastNode = h_lastNode[0];
    // h_lastNodeOfEachLevel[i+1]=lastNode;
}

/* now we have last nodes of each level stored in h_lastNodeOfEachLevel array*/


/* ---- process level 0 separately ------ */ 
int totalThreads = h_lastNodeOfEachLevel[0] - 0 + 1;
int totalBlocks = (totalThreads+1024-1)/1024;
firstNode = 0;
lastNode = h_lastNodeOfEachLevel[0];
activateLevel0<<<totalBlocks,1024>>>(firstNode,lastNode,V, 0, d_aid, d_activeness
, d_csrList, d_offset);
cudaDeviceSynchronize();



/* ---- process level 1 to (L-1) now  ------  */
firstNode = h_lastNodeOfEachLevel[0]+1;
lastNode = h_lastNodeOfEachLevel[1];

for(int i=1; i<L;i++){
    totalThreads = lastNode - firstNode + 1;
    totalBlocks = (totalThreads+1024-1)/1024;
    // call activate 
    activateRestOfTheLevels<<<totalBlocks, 1024>>>(firstNode, lastNode, i, 
    d_offset, d_csrList, d_aid, d_apr, d_activeness);
    cudaDeviceSynchronize();
    // call deactivate 
    deactivateRestOfTheLevels<<<totalBlocks, 1024>>>(firstNode, lastNode, i, 
    d_offset, d_csrList, d_aid, d_apr, d_activeness);
    cudaDeviceSynchronize();
    firstNode = lastNode+1;
    lastNode = h_lastNodeOfEachLevel[i+1];
}
    


/* Calculate number of active nodes on each level and store it in d_activeVertex*/
int count = 0;
firstNode = 0;
lastNode = h_lastNodeOfEachLevel[0];

for(int i =0; i<L ; i++){
  totalThreads = lastNode - firstNode + 1;
  totalBlocks = (totalThreads + 1024 - 1)/1024;
  countActiveNodes<<<totalBlocks, 1024>>>(i , firstNode, lastNode, 
  d_activeVertex, d_activeness);
  cudaDeviceSynchronize();
  firstNode = lastNode +1;
  lastNode = h_lastNodeOfEachLevel[i+1];
}
    
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);
   

     

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include "mpi.h"
using namespace std;

#define PIVOT_TAG 0
#define ARRAY_TAG 1
#define ARRAY_SIZE_TAG 2

void LThan(int *M, int &Size, int *NewM, int &SizeNewM, int *Send, int &SendSize, int P) {

	for (int t = 0; t < Size; ++t) {
		if (M[t] <= P) {
			NewM[SizeNewM] = M[t];
			++SizeNewM;
		}
		else {
			Send[SendSize] = M[t];
			++SendSize;
		}
	}
}

void GrThan(int *M, int &Size, int *NewM, int &SizeNewM, int *Send, int &SendSize, int P) {
	for (int t = 0; t < Size; ++t) {
		if (M[t] > P) {
			NewM[SizeNewM] = M[t];
			++SizeNewM;
		}
		else {
			Send[SendSize] = M[t];
			++SendSize;
		}
	}
}

void PSsort(int *M, int MSize, int RootP) {
	int PNum, PRanck, MNum, BSize, PMasSize, PMasNewSize, PMasSendSize, PMasRecvSize;
	int *PMas, *PMasNew, *PMasSend;
	MPI_Comm_size(MPI_COMM_WORLD, &PNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &PRanck);

	if (PNum == 1) {
		std::sort(M, M + MSize);
		return;
	}

	if (PRanck == 0) {
		BSize = MSize;
	}
	MPI_Bcast(&BSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MNum = BSize / PNum;

	int resSize = MNum * PNum;
	int diff = BSize - resSize;

	int *scounts = new int[PNum];
	for (int i = 0; i < PNum - 1; ++i)
		scounts[i] = MNum;
	scounts[PNum - 1] = MNum + diff;

	int *displs = new int[PNum];
	displs[0] = 0;
	for (int i = 1; i < PNum; ++i) {
		displs[i] = displs[i - 1] + scounts[i - 1];
	}
	PMasSize = scounts[PRanck];
	PMas = new int[PMasSize];
	MPI_Scatterv(M, scounts, displs, MPI_INT, PMas, scounts[PRanck], MPI_INT, 0, MPI_COMM_WORLD);
	delete[] scounts;
	delete[] displs;

	int countOfIterations = 0;
	while ((1 << countOfIterations) != PNum && (1 << countOfIterations) < PNum) {
		++countOfIterations;
	}


	int currentBit, stepMainOfCube, pairProcess, pivot;
	for (int i = countOfIterations - 1; i >= 0; --i) {

		currentBit = 1 << i;
		stepMainOfCube = 1 << (i + 1);
		bool isMainOfCube = false;
		
		pairProcess = PRanck | currentBit;
		if (PRanck == pairProcess)
			pairProcess -= currentBit;

		
		for (int j = 0; j < PNum; j += stepMainOfCube) {
			if (PRanck == j) {
				isMainOfCube = true;
				
				if (PMasSize != 0)
					pivot = PMas[PMasSize / 2];
				else
					pivot = 0;
				
				for (int t = PRanck + 1; t < PRanck + stepMainOfCube; ++t) {
					MPI_Send(&pivot, 1, MPI_INT, t, PIVOT_TAG, MPI_COMM_WORLD);
				}
				break;
			}
		}
		if (!isMainOfCube) {
			MPI_Status status;
			MPI_Recv(&pivot, 1, MPI_INT, MPI_ANY_SOURCE, PIVOT_TAG, MPI_COMM_WORLD, &status);
		}

		PMasNew = new int[PMasSize];
		PMasSend = new int[PMasSize];
		PMasNewSize = 0;
		PMasSendSize = 0;
		if (PRanck < pairProcess) {
			LThan(PMas, PMasSize, PMasNew, PMasNewSize, PMasSend
				, PMasSendSize, pivot);
		}
		else {
			GrThan(PMas, PMasSize, PMasNew, PMasNewSize, PMasSend
				, PMasSendSize, pivot);
		}

		delete[] PMas; 
		if (PRanck < pairProcess) {
			MPI_Send(&PMasSendSize, 1, MPI_INT, pairProcess, ARRAY_SIZE_TAG, MPI_COMM_WORLD);
			MPI_Send(PMasSend, PMasSendSize, MPI_INT, pairProcess, ARRAY_TAG, MPI_COMM_WORLD);

			delete[] PMasSend;
			MPI_Status status;
			MPI_Recv(&PMasRecvSize, 1, MPI_INT, pairProcess, ARRAY_SIZE_TAG, MPI_COMM_WORLD, &status);
			PMasSize = PMasNewSize + PMasRecvSize; 
			PMas = new int[PMasSize];
			memcpy(PMas, PMasNew, PMasNewSize * sizeof(int));

			delete[] PMasNew;

			MPI_Recv(PMas + PMasNewSize, PMasRecvSize, MPI_INT, pairProcess, ARRAY_TAG, MPI_COMM_WORLD, &status);
		}
		else {
			MPI_Status status;
			MPI_Recv(&PMasRecvSize, 1, MPI_INT, pairProcess, ARRAY_SIZE_TAG, MPI_COMM_WORLD, &status);
			
			PMasSize = PMasNewSize + PMasRecvSize; 
			PMas = new int[PMasSize];
			memcpy(PMas, PMasNew, PMasNewSize * sizeof(int));
			delete[] PMasNew;

			MPI_Recv(PMas + PMasNewSize, PMasRecvSize, MPI_INT, pairProcess, ARRAY_TAG, MPI_COMM_WORLD, &status);

			MPI_Send(&PMasSendSize, 1, MPI_INT, pairProcess, ARRAY_SIZE_TAG, MPI_COMM_WORLD);
			MPI_Send(PMasSend, PMasSendSize, MPI_INT, pairProcess, ARRAY_TAG, MPI_COMM_WORLD);

			delete[] PMasSend;
		}

	}

	
	std::sort(PMas, PMas + PMasSize);
	int *rcounts = new int[PNum];
	displs = new int[PNum];
	MPI_Gather(&PMasSize, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
	displs[0] = 0;
	for (int i = 1; i < PNum; ++i) {
		displs[i] = displs[i - 1] + rcounts[i - 1];
	}
	MPI_Gatherv(PMas, PMasSize, MPI_INT, M, rcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

	delete[] PMas;
	delete[] rcounts;
	delete[] displs;
}


int main(int argc, char* argv[]) {


	MPI_Init(&argc, &argv);
	int PRanck, PNum;
	MPI_Comm_rank(MPI_COMM_WORLD, &PRanck);
	MPI_Comm_size(MPI_COMM_WORLD, &PNum);
	int *M;
	int MSize;
	double start, finish;

	if (PRanck == 0) {
		ifstream input(argv[1]);
		input >> MSize;
		cout << "MassSize" << endl;
		cout << MSize << endl;
		M = new int[MSize];
		for (int i = 0; i < MSize; ++i) {
			input >> M[i];
		}
	}

	double interval, avg;
	avg = 0;
	for (int i = 0; i < 10; ++i) {
		std::random_shuffle(M, M + MSize);
		start = MPI_Wtime();
		PSsort(M, MSize, 0);
		finish = MPI_Wtime();
		interval = finish - start;
		avg += interval;
	}
	avg /= 10;
	delete[] M;
	MPI_Finalize();
	if (PRanck == 0)
	{
		cout << "AvgSize" << endl;
		cout << avg << endl;
	}
	return 0;
}
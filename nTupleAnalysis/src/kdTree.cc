
#include "ZZ4b/nTupleAnalysis/interface/kdTree.h"

namespace kdTree {

  int selecti(const int k, int *indx, int n, double *arr)
  // permutates indx[0..n-1] to make arr[indx[0..k-1] <= arr[indx[k] <= arr[indx[k+1..n-1]
  // the arr is not modified
  {
    int i,ia,ir,j,l,mid;
    double a;

    l=0;
    ir=n-1;
    for (;;) {
      if (ir <= l+1) {
	if (ir == l+1 && arr[indx[ir]] < arr[indx[l]])
	  SWAP(indx[l],indx[ir]);
	return indx[k];
      } else {
	mid=(l+ir) >> 1;
	SWAP(indx[mid],indx[l+1]);
	if (arr[indx[l]] > arr[indx[ir]]) SWAP(indx[l],indx[ir]);
	if (arr[indx[l+1]] > arr[indx[ir]]) SWAP(indx[l+1],indx[ir]);
	if (arr[indx[l]] > arr[indx[l+1]]) SWAP(indx[l],indx[l+1]);
	i=l+1;
	j=ir;
	ia = indx[l+1];
	a=arr[ia];
	for (;;) {
	  do i++; while (arr[indx[i]] < a);
	  do j--; while (arr[indx[j]] > a);
	  if (j < i) break;
	  SWAP(indx[i],indx[j]);
	}
	indx[l+1]=indx[j];
	indx[j]=ia;
	if (j >= k) ir=j-1;
	if (j <= k) l=i;
      }
    }
  }

}

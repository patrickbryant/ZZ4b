// -*- C++ -*-
#if !defined(kdTree_H)
#define kdTree_H

#include "ZZ4b/nTupleAnalysis/interface/pointbox.h"

#include <vector>
#include <iostream>

template<class T>
inline void SWAP(T &a, T &b)
{T dum=a; a=b; b=dum;}


namespace kdTree {


  template<int DIM> struct Boxnode : Box<DIM> {
    int mom, dau1, dau2, ptlo, pthi;
    Boxnode() {}
  Boxnode(Point<DIM> mylo, Point<DIM> myhi, int mymom, int myd1, int myd2, int myptlo, int mypthi) : 
    Box<DIM>(mylo, myhi), mom(mymom), dau1(myd1), dau2(myd2), ptlo(myptlo), pthi(mypthi)  {}
  };  


  template<int DIM> struct kdTree {
    static const double BIG;          // size of root box, value set below
    std::vector< Point<DIM> >& ptss;  // reference to teh vector of points in the KD tree
    int nboxes, npts;                 // number of boxes, numebr of points
    Boxnode<DIM>* boxes;              // The array of boxnodes that form the tree
    std::vector<int> ptindx, rptindx;      // index of points, and revse index
    double* coords;                   // point cooridantes rearranged contiguously
    kdTree(std::vector< Point<DIM> >&  pts); 
    ~kdTree() {delete [] boxes;}
    
    double disti(int jpt, int kpt);
    int locate(Point<DIM> pt);
    int locate(int jpt);
    int nearest(Point<DIM> pt);
    void nnearest(int jpt, int* nn, double* dn, int n);
    static void sift_down(double* heap, int* ndx, int nn); // Used by nnearest
    int locatenear(Point<DIM> pt, double r, int* list, int nmax);
  };


  template<int DIM> const double  kdTree<DIM>::BIG(1.0e99);
  

  int selecti(const int k, int *indx, int n, double *arr);
    
  
  template<int DIM> kdTree<DIM>::kdTree(std::vector< Point<DIM> > &pts) :
  ptss(pts), npts(pts.size()), ptindx(npts), rptindx(npts) {
    int ntmp,m,k,kk,j,nowtask,jbox,np,tmom,tdim,ptlo,pthi;
    int *hp;
    double *cp;
    int taskmom[50], taskdim[50];
    for (k=0; k<npts; k++) ptindx[k] = k;
    m = 1;
    for (ntmp = npts; ntmp; ntmp >>= 1) {
      m <<= 1;
    }
    nboxes = 2*npts - (m >> 1);
    if (m < nboxes) nboxes = m;
    nboxes--;
    boxes = new Boxnode<DIM>[nboxes];
    coords = new double[DIM*npts];
    for (j=0, kk=0; j<DIM; j++, kk += npts) {
      for (k=0; k<npts; k++) coords[kk+k] = pts[k].x[j];
    }
    Point<DIM> lo(-BIG,-BIG,-BIG,-BIG), hi(BIG,BIG,BIG,BIG);
    boxes[0] = Boxnode<DIM>(lo, hi, 0, 0, 0, 0, npts-1);
    jbox = 0;
    taskmom[1] = 0;
    taskdim[1] = 0;
    nowtask = 1;
    while (nowtask) {
      tmom = taskmom[nowtask];
      tdim = taskdim[nowtask--];
      ptlo = boxes[tmom].ptlo;
      pthi = boxes[tmom].pthi;
      hp = &ptindx[ptlo];
      cp = &coords[tdim*npts];
      np = pthi - ptlo + 1;
      kk = (np-1)/2;
      (void) selecti(kk,hp,np,cp);
      hi = boxes[tmom].hi;
      lo = boxes[tmom].lo;
      hi.x[tdim] = lo.x[tdim] = coords[tdim*npts + hp[kk]];
      boxes[++jbox] = Boxnode<DIM>(boxes[tmom].lo,hi,tmom,0,0,ptlo,ptlo+kk);
      boxes[++jbox] = Boxnode<DIM>(lo,boxes[tmom].hi,tmom,0,0,ptlo+kk+1,pthi);
      boxes[tmom].dau1 = jbox-1;
      boxes[tmom].dau2 = jbox;
      if (kk > 1) {
	taskmom[++nowtask] = jbox-1;
	taskdim[nowtask] = (tdim+1) % DIM;
      }
      if (np - kk > 3) {
	taskmom[++nowtask] = jbox;
	taskdim[nowtask] = (tdim+1) % DIM;
      }
    }
    for (j=0; j<npts; j++) rptindx[ptindx[j]] = j;
    delete [] coords;
  }


  template<int DIM> double kdTree<DIM>::disti(int jpt, int kpt) {
    if (jpt == kpt) return BIG;
    else return dist(ptss[jpt], ptss[kpt]);
  }

  template<int DIM> int kdTree<DIM>::locate(Point<DIM> pt) {
    int nb,d1,jdim;
    nb = jdim = 0;
    while (boxes[nb].dau1) {
      d1 = boxes[nb].dau1;
      if (pt.x[jdim] <= boxes[d1].hi.x[jdim]) nb=d1;
      else nb=boxes[nb].dau2;
      jdim = ++jdim % DIM;
    }
    return nb;
  }


  template<int DIM> int kdTree<DIM>::locate(int jpt) {
    int nb,d1,jh;
    jh = rptindx[jpt];
    nb = 0;
    while (boxes[nb].dau1) {
      d1 = boxes[nb].dau1;
      if (jh <= boxes[d1].pthi) nb=d1;
      else nb = boxes[nb].dau2;
    }
    return nb;
  }

  template<int DIM> int kdTree<DIM>::nearest(Point<DIM> pt) {
    int nrst = -1;
    int i,k,ntask;
    int task[50];
    double dnrst = BIG, d;
    std::cout << "dnrst is " << dnrst << std::endl;
    k = locate(pt);
    for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
      d = dist(ptss[ptindx[i]],pt);
      std::cout << "d from " << ptindx[i] << " is " << d << std::endl;
      if (d < dnrst) {
	nrst = ptindx[i];
	dnrst = d;
      }
    }
    task[1] = 0;
    ntask = 1;
    while (ntask) {
      k = task[ntask--];
      if (dist(boxes[k],pt) < dnrst) {
	if (boxes[k].dau1) {
	  task[++ntask] = boxes[k].dau1;
	  task[++ntask] = boxes[k].dau2;
	} else {
	  for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
	    d = dist(ptss[ptindx[i]],pt);
	    std::cout << "now d from " << ptindx[i] << " is " << d << std::endl;
	    if (d < dnrst) {
	      nrst = ptindx[i];
	      dnrst = d;
	    }
	  }
	}
      }
    }
    return nrst;
  }


  template<int DIM> void kdTree<DIM>::nnearest(int jpt, int *nn, double *dn, int n)
    {
      int i,k,ntask,kp;
      int task[50];
      double d;
      if (n > npts-1) throw("too many neighbors requested");
      for (i=0; i<n; i++) dn[i] = BIG;
      kp = boxes[locate(jpt)].mom;
      while (boxes[kp].pthi - boxes[kp].ptlo < n) kp = boxes[kp].mom;
      for (i=boxes[kp].ptlo; i<=boxes[kp].pthi; i++) {
	if (jpt == ptindx[i]) continue;
	d = disti(ptindx[i],jpt);
	if (d < dn[0]) {
	  dn[0] = d;
	  nn[0] = ptindx[i];
	  if (n>1) sift_down(dn,nn,n);
	}
      }
      task[1] = 0;
      ntask = 1;
      while (ntask) {
	k = task[ntask--];
	if (k == kp) continue;
	if (dist(boxes[k],ptss[jpt]) < dn[0]) {
	  if (boxes[k].dau1) {
	    task[++ntask] = boxes[k].dau1;
	    task[++ntask] = boxes[k].dau2;
	  } else {
	    for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
	      d = disti(ptindx[i],jpt);
	      if (d < dn[0]) {
		dn[0] = d;
		nn[0] = ptindx[i];
		if (n>1) sift_down(dn,nn,n);
	      }
	    }
	  }
	}
      }
      return;
    }


  template<int DIM> void kdTree<DIM>::sift_down(double *heap, int *ndx, int nn) {
    int n = nn - 1;
    int j,jold,ia;
    double a;
    a = heap[0];
    ia = ndx[0];
    jold = 0;
    j = 1;
    while (j <= n) {
      if (j < n && heap[j] < heap[j+1]) j++;
      if (a >= heap[j]) break;
      heap[jold] = heap[j];
      ndx[jold] = ndx[j];
      jold = j;
      j = 2*j + 1;
    }
    heap[jold] = a;
    ndx[jold] = ia;
  }


  template<int DIM>
    int kdTree<DIM>::locatenear(Point<DIM> pt, double r, int *list, int nmax) {
    int k,i,nb,nbold,nret,ntask,jdim,d1,d2;
    int task[50];
    nb = jdim = nret = 0;
    if (r < 0.0) throw("radius must be nonnegative");
    while (boxes[nb].dau1) {
      nbold = nb;
      d1 = boxes[nb].dau1;
      d2 = boxes[nb].dau2;
      if (pt.x[jdim] + r <= boxes[d1].hi.x[jdim]) nb = d1;
      else if (pt.x[jdim] - r >= boxes[d2].lo.x[jdim]) nb = d2;
      jdim = ++jdim % DIM;
      if (nb == nbold) break;
    }
    task[1] = nb;
    ntask = 1;
    while (ntask) {
      k = task[ntask--];
      if (dist(boxes[k],pt) > r) continue;
      if (boxes[k].dau1) {
	task[++ntask] = boxes[k].dau1;
	task[++ntask] = boxes[k].dau2;
      } else {
	for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
	  if (dist(ptss[ptindx[i]],pt) <= r && nret < nmax)
	    list[nret++] = ptindx[i];
	  if (nret == nmax) return nmax;
	}
      }
    }
    return nret;
  }

}
#endif // kdTree_H

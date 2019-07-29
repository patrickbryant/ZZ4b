// -*- C++ -*-
#if !defined(pointbox_H)
#define pointbox_H
#include <math.h>


namespace kdTree {

  //
  //  Point 
  //
  template<int DIM> struct Point {

    double x[DIM];
    Point(const Point& p){
      for(int i=0; i<DIM; ++i) x[i] = p.x[i];
    }
    
    Point& operator = (const Point& p) {
      for(int i=0; i<DIM; ++i) x[i] = p.x[i];
      return *this;
    }
      
    bool operator == (const Point& p) const {
      for(int i=0; i<DIM; ++i) if(x[i] != p.x[i]) return false;
      return true;
    }
     
    Point(double x0=0, double x1=0, double x2=0, double x3=0){
      x[0] = x0;
      if(DIM > 1) x[1] = x1;
      if(DIM > 2) x[2] = x2;
      if(DIM > 3) x[3] = x3;
      if(DIM > 4) throw("Point not implemented for DIM > 4");
    }

  };

  //
  //  Box
  //
  template<int DIM> struct Box {
    Point<DIM> lo, hi;  // diagonally opposite corners
    Box() {}
    Box(const Point<DIM>& mylo, const Point<DIM>& myhi) : lo(mylo), hi(myhi) {}
  };  



  //
  //  Distances
  //
  template<int DIM> double dist(const Point<DIM>& p, const Point<DIM>& q){
    double dd = 0.0;
    for(int j=0; j<DIM; ++j) dd += pow( (q.x[j]-p.x[j]) ,2);
    return sqrt(dd);
  }


  template<int DIM> double dist(const Box<DIM>& b, const Point<DIM>& p){
    double dd = 0.0;
    for(int i=0; i<DIM; ++i){
      if(p.x[i] < b.lo.x[i]) dd += pow( (p.x[i]-b.lo.x[i]) ,2);
      if(p.x[i] > b.hi.x[i]) dd += pow( (p.x[i]-b.hi.x[i]) ,2);
    }
    return sqrt(dd);
  }


  
    
  

}
#endif // pointbox_H

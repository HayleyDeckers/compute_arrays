#include "../include/compute_arrays.hpp"
#include <iostream>
using namespace compute_arrays;

/*
  This example computes the derivative of `sin(x)` on [0,3.14] using both the [forward
  finite difference and central finite difference](https://en.wikipedia.org/wiki/Finite_difference#Forward.2C_backward.2C_and_central_differences) approximation.
   It then prints the mean absolute error for both methods as a function of the grid
  spacing `dx`.
   Because the output is only the sum of an expression, no arrays are allocated. Compare the maximum memory
  used as you increase the upper bound on n_points to observer this.
*/

int main(){
  for(std::size_t n_points = (1<<10); n_points < (1ul<<26); n_points *= 2){
    //create a grid on [0,3.14] using n_points.
    //Range(0,1,6) -> [0,0.2,0.4,0.6,0.8,1.0]
    // this does not actually create an array but dynamically computes when indexed.
    auto x = Range<double>(0,3.14,n_points);
    //taking the sin of x does not create an array either...
    auto y = sin(x);

    //print the stepsize
    std::cout << x.delta() << " ";

    {//first-order forward-difference
      //using array indices this scheme is d/dx[0,1,2,3] = ([1,2,3,4] - [0,1,2,3])/dx
      //we can take subsections of an expression using the make_view functions.
      auto dydx = (make_view_from(y,1)-make_view_to(y,x.size()-2))/x.delta();
      //d/dx sin(x) = cos(x). We do not have a result for the last point of x so we take another view.
      auto expected_result = cos(make_view_to(x,x.size()-2));
      //declare the pointwise error of our finite difference approximation.
      auto pointwise_error = abs(dydx-expected_result);
      //now compute the mean error. This is the first line of code that actually
      // _computes_ anything. It performs all the steps above, vectorized and
      // multithreaded and returns the sum of all elements.
      double mean_error = reduce::sum(pointwise_error)/pointwise_error.size();
      std::cout << mean_error << " ";
    }
    {//first-order central-difference
      //this scheme looks like d/dx[1,2,3] = ([2,3,4]-[0,1,2])/(2*dx)
      auto dydx = (make_view_from(y,2)-make_view_to(y,x.size()-3))/(2*x.delta());
      //here we use make_view to drop the first and last element.
      auto expected_result = cos(make_view(x,1,x.size()-2));
      auto pointwise_error = abs(dydx-expected_result);
      double mean_error = reduce::sum(pointwise_error)/pointwise_error.size();
      std::cout << mean_error << std::endl;
    }
  }
}

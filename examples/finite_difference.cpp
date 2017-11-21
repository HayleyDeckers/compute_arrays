#include "../compute_arrays.hpp"
#include <iostream>
using namespace compute_arrays;

int main(){
  for(int n_points = 8; n_points < 3300000; n_points *= 2){
    auto x = Range<double>(0,3.14,n_points);
    std::cout << x.delta() << " ";
    auto y = sin(x);
    //[0,1,2,3,4]
    {//first-order forward-difference
      //[1,2,3,4] - [0,1,2,3]
      auto dydx = (make_view_from(y,1)-make_view_to(y,x.size()-2))/x.delta();
      auto expected_result = cos(make_view_to(x,x.size()-2));
      auto pointwise_error = abs(dydx-expected_result);
      double mean_error = reduce::sum(pointwise_error)/pointwise_error.size();
      std::cout << mean_error << " ";
    }
    {//first-order central-difference
      //[2,3,4]-[0,1,2]
      //y[{2,y.size()-1}] - y[{0,y.size()-2}]
      auto dydx = (make_view_from(y,2)-make_view_to(y,x.size()-3))/(2*x.delta());
      auto expected_result = cos(make_view(x,1,x.size()-2));
      auto pointwise_error = abs(dydx-expected_result);
      double mean_error = reduce::sum(pointwise_error)/pointwise_error.size();
      std::cout << mean_error << std::endl;
    }
  }
}

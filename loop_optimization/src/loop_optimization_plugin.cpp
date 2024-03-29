#include "loop_optimization_plugin.h"
#include "loop_optimization_node.h"
#include <pluginlib/class_list_macros.h>
#include <boost/thread.hpp>

loop_optimization_plugin::loop_optimization_plugin()
{

}

void loop_optimization_plugin::onInit(){
  NODELET_INFO("loop_optimization_plugin - %s", __FUNCTION__);

  boost::shared_ptr<boost::thread> spinThread3;
  spinThread3 = boost::shared_ptr< boost::thread >
              (new boost::thread(&mainOptimizationFunction));
}

PLUGINLIB_EXPORT_CLASS(loop_optimization_plugin, nodelet::Nodelet)

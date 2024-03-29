#include "loop_detection_plugin.h"
#include "loop_detection.h"
#include <pluginlib/class_list_macros.h>
#include <boost/thread.hpp>

loop_detection_plugin::loop_detection_plugin()
{

}

void loop_detection_plugin::onInit(){
  NODELET_INFO("loop_detection_plugin - %s", __FUNCTION__);

  boost::shared_ptr<boost::thread> spinThread1;
  spinThread1 = boost::shared_ptr< boost::thread >
              (new boost::thread(&mainLCFunction));
}

PLUGINLIB_EXPORT_CLASS(loop_detection_plugin, nodelet::Nodelet)

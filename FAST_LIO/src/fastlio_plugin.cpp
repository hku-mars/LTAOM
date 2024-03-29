#include "fastlio_plugin.h"
#include "laserMapping.h"

#include <pluginlib/class_list_macros.h>
#include <boost/thread.hpp>

fastlio_plugin::fastlio_plugin()
{

}

void fastlio_plugin::onInit(){
  NODELET_INFO("fastlio_plugin - %s", __FUNCTION__);

  boost::shared_ptr<boost::thread> spinThread2;
  spinThread2 = boost::shared_ptr< boost::thread >
              (new boost::thread(&mainLIOFunction));
}

PLUGINLIB_EXPORT_CLASS(fastlio_plugin, nodelet::Nodelet)


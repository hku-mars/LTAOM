#ifndef LOOP_DETECTION_PLUGIN_H
#define LOOP_DETECTION_PLUGIN_H
#include <nodelet/nodelet.h>

class loop_detection_plugin: public nodelet::Nodelet
{
public:
  loop_detection_plugin();

  virtual void onInit();
};

#endif // LOOP_DETECTION_PLUGIN_H

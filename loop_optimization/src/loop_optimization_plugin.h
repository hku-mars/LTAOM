#ifndef LOOP_OPTIMIZATION_PLUGIN_H
#define LOOP_OPTIMIZATION_PLUGIN_H
#include <nodelet/nodelet.h>

class loop_optimization_plugin:public nodelet::Nodelet
{
public:
  loop_optimization_plugin();

  virtual void onInit();
};

#endif // LOOP_OPTIMIZATION_PLUGIN_H

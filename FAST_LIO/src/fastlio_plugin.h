#ifndef FASTLIO_PLUGIN_H
#define FASTLIO_PLUGIN_H
#include <nodelet/nodelet.h>

class fastlio_plugin: public nodelet::Nodelet
{
public:
  fastlio_plugin();

  virtual void onInit();

};

#endif // FASTLIO_PLUGIN_H

#include "graphbuilderadapter.h"

#include "Msnhnet/3rdparty/yaml-cpp/parser.h"  // IWYU pragma: keep

namespace YAML {
class GraphBuilderInterface;

void* BuildGraphOfNextDocument(Parser& parser,
                               GraphBuilderInterface& graphBuilder) {
  GraphBuilderAdapter eventHandler(graphBuilder);
  if (parser.HandleNextDocument(eventHandler)) {
    return eventHandler.RootNode();
  } else {
    return nullptr;
  }
}
}

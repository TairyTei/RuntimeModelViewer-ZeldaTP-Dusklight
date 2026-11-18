#include "app/inspector_service.hpp"

#include <vector>

#include "shared/core.hpp"
#include "preview/preview.hpp"

namespace mi {

void inspect_model(uint64_t key) {
    select_model(key);
    preview_open();
}

bool run_selector(size_t index) {
    const std::vector<ModelInfo> models = registry_snapshot();
    const std::vector<Selector>& registered = selectors();
    if (index >= registered.size()) return false;

    PickContext context;
    context.cam = camera();
    context.models = &models;
    uint64_t key = 0;
    if (!registered[index].pick(context, &key)) return false;
    inspect_model(key);
    return true;
}

}  // namespace mi

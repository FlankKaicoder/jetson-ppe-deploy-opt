#include <NvInferRuntime.h>

#include <dlfcn.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using InitFunction = bool (*)();
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: exp16_plugin_probe PLUGIN_SO");
        }
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }
        auto init = reinterpret_cast<InitFunction>(dlsym(library, "ppeInitPlugin"));
        if (init == nullptr) {
            throw std::runtime_error("dlsym ppeInitPlugin failed");
        }
        if (!init()) {
            throw std::runtime_error("plugin registration failed");
        }
        auto* creator = getPluginRegistry()->getCreator(
            "PpeYoloDecodeCompact", "1", "com.flankkaicoder.ppe");
        if (creator == nullptr) {
            throw std::runtime_error("registered creator lookup failed");
        }
        std::cout << "plugin_probe=PASS\n"
                  << "name=PpeYoloDecodeCompact\n"
                  << "version=1\n"
                  << "namespace=com.flankkaicoder.ppe\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "plugin_probe=FAIL\nerror=" << error.what() << '\n';
        return 1;
    }
}

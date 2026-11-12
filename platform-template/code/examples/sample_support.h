#pragma once
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

inline void check(int rc) {
    if (rc)
        throw std::runtime_error("ACL error " + std::to_string(rc));
}
inline std::vector<char> read_file(const std::string& name, size_t size) {
    std::ifstream file(name, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(size))
        throw std::runtime_error("input size: " + name);
    std::vector<char> data(size);
    file.seekg(0);
    if (size && !file.read(data.data(), size))
        throw std::runtime_error("read: " + name);
    return data;
}
struct Tensor {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
    aclScalar* scalar = nullptr;
    size_t bytes;
    std::string file;
    std::vector<char> host;
    Tensor(std::vector<int64_t> shape, aclDataType type, size_t size, const char* name, bool input,
           bool is_scalar, const std::string& root)
        : bytes(size), file(name) {
        host = input ? read_file(root + "/input/" + file, size) : std::vector<char>(size);
        if (is_scalar) {
            scalar = aclCreateScalar(host.data(), type);
            if (!scalar)
                throw std::runtime_error("scalar creation failed");
            return;
        }
        check(aclrtMalloc(&device, std::max(size_t(1), size), ACL_MEM_MALLOC_HUGE_FIRST));
        if (size)
            check(aclrtMemcpy(device, size, host.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i)
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), device);
        if (!tensor)
            throw std::runtime_error("tensor creation failed");
    }
    void save(const std::string& root) {
        if (bytes)
            check(aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream out(root + "/output/" + file, std::ios::binary);
        if (!out || (bytes && !out.write(host.data(), bytes)))
            throw std::runtime_error("write: " + file);
    }
    ~Tensor() {
        if (scalar)
            aclDestroyScalar(scalar);
        if (tensor)
            aclDestroyTensor(tensor);
        if (device)
            aclrtFree(device);
    }
};
struct TensorList {
    aclTensorList* list;
    TensorList(std::initializer_list<aclTensor*> values) {
        std::vector<aclTensor*> items(values);
        list = aclCreateTensorList(items.data(), items.size());
        if (!list)
            throw std::runtime_error("tensor list creation failed");
    }
    ~TensorList() {
        aclDestroyTensorList(list);
    }
};

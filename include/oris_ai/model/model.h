/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>  // GLOG for logging

namespace oris_ai {

class Model {
  public:
    /**
     * @brief Constructor to initialize a Model with a given name.
     */
    Model(const std::string& model_name) : model_name_(model_name) {}

    virtual ~Model() = default;

    /**
     * @brief Pure virtual function to open the model.
     * Each derived class (e.g., Yolov8n) must implement this function.
     */
    virtual bool Open(const std::string& model_path) = 0;

  protected:
    std::string model_name_;  // The name of the model
};

/**
 * @brief Factory function to create a model based on the model name.
 */
std::unique_ptr<Model> CreateModel(const std::string& model_name);

}  // namespace oris_ai

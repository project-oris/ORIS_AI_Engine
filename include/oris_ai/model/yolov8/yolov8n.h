/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/model/model.h"

namespace oris_ai {

class Yolov8n : public Model {
  public:
    /**
     * @brief Constructor to initialize Yolov8n model.
     */
    Yolov8n() : Model("yolov8n") {}

    /**
     * @brief Opens the YOLOv8n model
     * 
     * @param model_path The path to the model file.
     * @return bool indicating success or failure.
     */
    bool Open(const std::string& model_path) override;

  private:
    bool ParsingModel(std::fstream& input);
};

}  // namespace oris_ai

//
// Created by 郑文鑫 on 2019/12/6.
//

#include "Conv2D.h"
peterzheng::model::Conv2D::Conv2D(
    const peterzheng::matrix::matrix<float> &filter,
    const peterzheng::matrix::matrix<float> &x)
    : filter(filter), x(x) {}
void peterzheng::model::Conv2D::run() {
  // direct do conv operation
}
void peterzheng::model::Conv2D::compile() {}
float peterzheng::model::Conv2D::loss() { return 0; }

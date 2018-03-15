#include "FullSystem/Frame.h"

namespace fdso {

Frame::Frame(FrameHessian* frame)
{
  update=true;
  lastUpdate=update;
  id = frame->shell->id;
  frameID = frame->frameID;
  camToWorld = frame->shell->camToWorld;
  camToWorldOpti = frame->shell->camToWorld;
  timestamp=frame->shell->timestamp;

  for (auto &fea : frame->_features)
  {
    fea->_host = this;
    _features.push_back(fea);
  }

  _bow_vec = frame->_bow_vec;
  _feature_vec = frame->_feature_vec;
}
/**
 * { item_description }
 * 获取当前帧的连接
 */
set<Frame*> Frame::GetConnectedKeyFrames()
{
  set<Frame*> connectedFrames;
  for (auto &rel : mPoseRel)
    connectedFrames.insert(rel.first);
  return connectedFrames;
}

void Frame::release()
{
  for (int i = 0; i < _features.size(); i++)
    delete _features[i];
  _features.clear();
}

}

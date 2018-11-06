#include "ConnectedComponentsWholeSlideFilter.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "core/PathologyEnums.h"
#include <set>
#include <iostream>

ConnectedComponentsWholeSlideFilter::DisjointSet::DisjointSet()
{
  m_numElements = 0;
  m_numSets = 0;
}

ConnectedComponentsWholeSlideFilter::DisjointSet::DisjointSet(int count)
{
  m_numElements = 0;
  m_numSets = 0;
  addElements(count);
}

ConnectedComponentsWholeSlideFilter::DisjointSet::DisjointSet(const DisjointSet & s)
{
  this->m_numElements = s.m_numElements;
  this->m_numSets = s.m_numSets;

  // Copy nodes
  m_nodes.resize(m_numElements);
  for (int i = 0; i < m_numElements; ++i)
    m_nodes[i] = new Node(*s.m_nodes[i]);

  // Update parent pointers to point to newly created nodes rather than the old ones
  for (int i = 0; i < m_numElements; ++i)
    if (m_nodes[i]->parent != NULL)
      m_nodes[i]->parent = m_nodes[s.m_nodes[i]->parent->index];
}

ConnectedComponentsWholeSlideFilter::DisjointSet::~DisjointSet()
{
  for (int i = 0; i < m_numElements; ++i)
    delete m_nodes[i];
  m_nodes.clear();
  m_numElements = 0;
  m_numSets = 0;
}

// Note: some internal data is modified for optimization even though this method is consant.
int ConnectedComponentsWholeSlideFilter::DisjointSet::findSet(int elementId) const
{
  Node* curNode;

  // Find the root element that represents the set which `elementId` belongs to
  curNode = m_nodes[elementId];
  while (curNode->parent != NULL)
    curNode = curNode->parent;
  Node* root = curNode;

  // Walk to the root, updating the parents of `elementId`. Make those elements the direct
  // children of `root`. This optimizes the tree for future FindSet invokations.
  curNode = m_nodes[elementId];
  while (curNode != root)
  {
    Node* next = curNode->parent;
    curNode->parent = root;
    curNode = next;
  }

  return root->index;
}

void ConnectedComponentsWholeSlideFilter::DisjointSet::set_union(int setId1, int setId2)
{
  if (setId1 == setId2)
    return; // already unioned

  Node* set1 = m_nodes[setId1];
  Node* set2 = m_nodes[setId2];

  // Determine which node representing a set has a higher rank. The node with the higher rank is
  // likely to have a bigger subtree so in order to better balance the tree representing the
  // union, the node with the higher rank is made the parent of the one with the lower rank and
  // not the other way around.
  if (set1->rank > set2->rank)
    set2->parent = set1;
  else if (set1->rank < set2->rank)
    set1->parent = set2;
  else // set1->rank == set2->rank
  {
    set2->parent = set1;
    ++set1->rank; // update rank
  }

  // Since two sets have fused into one, there is now one less set so update the set count.
  --m_numSets;
}

void ConnectedComponentsWholeSlideFilter::DisjointSet::addElements(int numToAdd)
{

  // insert and initialize the specified number of element nodes to the end of the `m_nodes` array
  m_nodes.insert(m_nodes.end(), numToAdd, (Node*)NULL);
  for (int i = m_numElements; i < m_numElements + numToAdd; ++i)
  {
    m_nodes[i] = new Node();
    m_nodes[i]->parent = NULL;
    m_nodes[i]->index = i;
    m_nodes[i]->rank = 0;
  }

  // update element and set counts
  m_numElements += numToAdd;
  m_numSets += numToAdd;
}

ConnectedComponentsWholeSlideFilter::ConnectedComponentsWholeSlideFilter() :
_monitor(NULL),
_processedLevel(0),
_outPath(""),
_threshold(0.5)
{

}

ConnectedComponentsWholeSlideFilter::~ConnectedComponentsWholeSlideFilter() {
}

void ConnectedComponentsWholeSlideFilter::setInput(const std::shared_ptr<MultiResolutionImage>& input) {
  _input = input;
}

void ConnectedComponentsWholeSlideFilter::setOutput(const std::string& outPath) {
  _outPath = outPath;
}

void ConnectedComponentsWholeSlideFilter::setProcessedLevel(const unsigned int processedLevel) {
  _processedLevel = processedLevel;
}

unsigned int ConnectedComponentsWholeSlideFilter::getProcessedLevel() {
  return _processedLevel;
}

void ConnectedComponentsWholeSlideFilter::setProgressMonitor(ProgressMonitor* progressMonitor) {
  _monitor = progressMonitor;
}

ProgressMonitor* ConnectedComponentsWholeSlideFilter::getProgressMonitor() {
  return _monitor;
}

void ConnectedComponentsWholeSlideFilter::setThreshold(const float& threshold) {
  this->_threshold = threshold;
}

float ConnectedComponentsWholeSlideFilter::getThreshold() {
  return _threshold;
}

bool ConnectedComponentsWholeSlideFilter::process() const {
  std::shared_ptr<MultiResolutionImage> img = _input.lock();
  std::vector<unsigned long long> dims = img->getLevelDimensions(this->_processedLevel);
  double downsample = img->getLevelDownsample(this->_processedLevel);

  MultiResolutionImageWriter writer;
  writer.setColorType(pathology::ColorType::Monochrome);
  writer.setCompression(pathology::Compression::LZW);
  writer.setDataType(pathology::DataType::UInt32);
  writer.setInterpolation(pathology::Interpolation::NearestNeighbor);
  writer.setTileSize(512);
  std::vector<double> spacing = img->getSpacing();
  if (!spacing.empty()) {
    spacing[0] *= downsample;
    spacing[1] *= downsample;
    writer.setSpacing(spacing);
  }
  if (writer.openFile(_outPath) != 0) {
    std::cerr << "ERROR: Could not open file for writing" << std::endl;
    return false;
  }
  writer.setProgressMonitor(_monitor);
  writer.writeImageInformation(dims[0], dims[1]);

  DisjointSet dset;
  dset.addElements(1);
  unsigned int* buffer_t_x = new unsigned int[512];
  unsigned int* buffer_t_y = new unsigned int[512 * dims[0]];
  float* tile = new float[512 * 512];
  unsigned int* label_tile = new unsigned int[512 * 512];
  std::fill(label_tile, label_tile + 512 * 512, 0);
  std::fill(buffer_t_x, buffer_t_x + 512, 0);
  std::fill(buffer_t_y, buffer_t_y + 512 * dims[0], 0);

  bool firstPass = true;
  std::vector<unsigned int> setIdToFinalLabel;
  for (unsigned int pass = 0; pass < 2; ++pass) {
    std::fill(buffer_t_y, buffer_t_y + 512 * dims[0], 0);
    unsigned int curLabel = 0;
    for (unsigned long long t_y = 0; t_y < dims[1]; t_y += 512) {
      std::fill(buffer_t_x, buffer_t_x + 512, 0);
      for (unsigned long long t_x = 0; t_x < dims[0]; t_x += 512) {
        img->getRawRegion<float>(static_cast<unsigned long long>(t_x*downsample), static_cast<unsigned long long>(t_y*downsample), 512, 512, this->_processedLevel, tile);
        std::fill(label_tile, label_tile + 512 * 512, 0);
        for (int y = 0; y < 512; ++y) {
          for (int x = 0; x < 512; ++x) {
            unsigned int curVal = tile[y * 512 + x] > _threshold;
            if (curVal == 1) {
              unsigned int leftVal = 0;
              unsigned int topVal = 0;
              if (x == 0) {
                leftVal = buffer_t_x[y];
              }
              else {
                leftVal = label_tile[y * 512 + x - 1];
              }
              if (y == 0) {
                topVal = buffer_t_y[t_x + x];
              }
              else {
                topVal = label_tile[(y - 1) * 512 + x];
              }
              if (leftVal == 0 && topVal == 0) {
                dset.addElements(1);
                curLabel++;
                if (firstPass) {
                  label_tile[y * 512 + x] = curLabel;
                }
                else {
                  label_tile[y * 512 + x] = setIdToFinalLabel[dset.findSet(curLabel)];
                }
              }
              else {
                unsigned int minLabel = std::min(leftVal, topVal) > 0 ? std::min(leftVal, topVal) : std::max(leftVal, topVal);
                label_tile[y * 512 + x] = minLabel;
                if (firstPass) {
                  if (topVal > 0 && leftVal > 0) {
                    dset.set_union(dset.findSet(leftVal), dset.findSet(topVal));
                  }
                }
              }
            }
            if (x == 511) {
              buffer_t_x[y] = label_tile[y * 512 + x];
            }
            if (y == 511) {
              buffer_t_y[t_x + x] = label_tile[y * 512 + x];
            }
          }
        }
        if (!firstPass) {
          writer.writeBaseImagePart(reinterpret_cast<void*>(label_tile));
        }
      }
    }
    firstPass = false;
    std::set<unsigned int> set_indices;
    setIdToFinalLabel.resize(dset.numElements());
    for (unsigned int i = 0; i < dset.numElements(); ++i) {
      set_indices.insert(dset.findSet(i));
    }
    int label = 0;
    for (std::set<unsigned int>::const_iterator it = set_indices.begin(); it != set_indices.end(); ++it) {
      setIdToFinalLabel[*it] = label;
      ++label;
    }
  }

  writer.finishImage();
  delete[] buffer_t_x;
  delete[] buffer_t_y;
  delete[] tile;
  delete[] label_tile;
  return true;
}
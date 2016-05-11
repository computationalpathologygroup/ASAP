#ifndef _ConnectedComponentsWholeSlideFilter
#define _ConnectedComponentsWholeSlideFilter

#include "wholeslidefilters_export.h"
#include <string>
#include <vector>
#include <memory>

class MultiResolutionImage;
class ProgressMonitor;

class WHOLESLIDEFILTERS_EXPORT ConnectedComponentsWholeSlideFilter {

private:
  std::weak_ptr<MultiResolutionImage> _input;
  ProgressMonitor* _monitor;
  unsigned int _processedLevel;
  std::string _outPath;
  float _threshold;

  class DisjointSet
  {
  public:

    // Create an empty DisjointSets data structure
    DisjointSet();
    // Create a DisjointSets data structure with a specified number of elements (with element id's from 0 to count-1)
    DisjointSet(int count);
    // Copy constructor
    DisjointSet(const DisjointSet & s);
    // Destructor
    ~DisjointSet();

    // Find the set identifier that an element currently belongs to.
    // Note: some internal data is modified for optimization even though this method is consant.
    int findSet(int element) const;
    // Combine two sets into one. All elements in those two sets will share the same set id that can be gotten using FindSet.
    void set_union(int setId1, int setId2);
    // Add a specified number of elements to the DisjointSets data structure. The element id's of the new elements are numbered
    // consequitively starting with the first never-before-used elementId.
    void addElements(int numToAdd);
    // Returns the number of elements currently in the DisjointSets data structure.
    inline int numElements() const { return m_numElements; }
    // Returns the number of sets currently in the DisjointSets data structure.
    inline int numSets() const { return m_numSets; }

  private:

    // Internal Node data structure used for representing an element
    struct Node
    {
      int rank; // This roughly represent the max height of the node in its subtree
      int index; // The index of the element the node represents
      Node* parent; // The parent node of the node
    };

    int m_numElements; // the number of elements currently in the DisjointSets data structure.
    int m_numSets; // the number of sets currently in the DisjointSets data structure.
    std::vector<Node*> m_nodes; // the list of nodes representing the elements
  };

public:
  ConnectedComponentsWholeSlideFilter();
  virtual ~ConnectedComponentsWholeSlideFilter();

  void setInput(const std::shared_ptr<MultiResolutionImage>& input);
  void setProcessedLevel(const unsigned int processedLevel);
  unsigned int getProcessedLevel();
  void setProgressMonitor(ProgressMonitor* progressMonitor);
  void setThreshold(const float& threshold);
  float getThreshold();
  ProgressMonitor* getProgressMonitor();
  bool process() const;
  void setOutput(const std::string& outPath);

};

#endif
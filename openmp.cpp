// openmp.cpp
// A single‐file OpenMP parallel implementation of Random Forest with AUC and accuracy
// Uses std::shuffle (instead of deprecated random_shuffle)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <functional>
#include <memory>
#include <random>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Use thread_local random engine (each thread gets its own instance).
thread_local std::mt19937 rng(std::random_device{}());

// ---------------- Utility Functions ----------------

vector<string> splitBySpace(const string &sentence) {
    istringstream iss(sentence);
    return vector<string>{istream_iterator<string>(iss), istream_iterator<string>()};
}

void writeDataToCSV(const vector<double> &results, const vector<int>& target, const string &filename, bool train) {
    ofstream out(filename);
    if (out.is_open()) {
        out << "id,label";
        if (train) { out << ",real\n"; } else { out << "\n"; }
        for (size_t i = 0; i < results.size(); i++) {
            out << i << "," << results[i];
            if (train && i < target.size()) {
                out << "," << target[i] << "\n";
            } else {
                out << "\n";
            }
        }
        out.close();
    } else {
        cout << "Write File failed" << endl;
    }
}

// ---------------- Metric Functions ----------------

double computeAUC(const vector<double>& scores, const vector<int>& labels) {
    int n = scores.size();
    vector<pair<double, int>> arr;
    for (int i = 0; i < n; i++) {
        arr.push_back({scores[i], labels[i]});
    }
    sort(arr.begin(), arr.end(), [](auto &a, auto &b) {
        return a.first < b.first;
    });
    int posCount = 0, negCount = 0;
    for (auto &p : arr) {
        if (p.second == 1) posCount++;
        else negCount++;
    }
    if (posCount == 0 || negCount == 0)
        return 0.5;
    double rankSum = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i].second == 1)
            rankSum += (i + 1);
    }
    double auc = (rankSum - posCount * (posCount + 1) / 2.0) / (posCount * negCount);
    return auc;
}

double computeAccuracy(const vector<double>& scores, const vector<int>& labels) {
    int correct = 0;
    int n = scores.size();
    for (int i = 0; i < n; i++) {
        int pred = (scores[i] >= 0.5) ? 1 : 0;
        if (pred == labels[i])
            correct++;
    }
    return correct / (double)n;
}

// ---------------- Data Class ----------------

class Data {
private:
    vector<vector<double>> features;
    vector<int> target;
    int featureSize;
    bool isTrain;
    vector<int> samplesVec;
public:
    Data(bool isTrain = true, int size = 1000, int featureSize = 201)
      : featureSize(featureSize), isTrain(isTrain) {
        features.reserve(size);
        if(isTrain)
            target.reserve(size);
        for (int i = 0; i < size; i++) {
            samplesVec.push_back(i);
        }
    }
    
    void read(const string &filename) {
        ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            cout << "Failed to open file: " << filename << endl;
            return;
        }
        string line;
        int startIndex = isTrain ? 1 : 0;
        while(getline(inputFile, line)) {
            if(line.empty()) continue;
            auto tokens = splitBySpace(line);
            vector<double> sample(featureSize, 0);
            if(isTrain && !tokens.empty())
                target.push_back(atoi(tokens[0].c_str()));
            for (size_t i = startIndex; i < tokens.size(); i++) {
                size_t pos = tokens[i].find(":");
                if(pos != string::npos) {
                    int key = atoi(tokens[i].substr(0, pos).c_str());
                    double value = atof(tokens[i].substr(pos + 1).c_str());
                    if(key < featureSize)
                        sample[key] = value;
                }
            }
            features.push_back(sample);
        }
        inputFile.close();
    }
    
    double readFeature(int sampleIndex, int featureIndex) const {
        return features[sampleIndex][featureIndex];
    }
    
    int readTarget(int sampleIndex) const {
        return target[sampleIndex];
    }
    
    int getSampleSize() const {
        return features.size();
    }
    
    int getFeatureSize() const {
        return featureSize;
    }
    
    const vector<int>& getSamplesVec() const {
        return samplesVec;
    }
    
    // Use std::shuffle instead of random_shuffle.
    vector<int> generateSample(int num) const {
        vector<int> samples = samplesVec;
        if(num == -1 || num >= (int)samples.size())
            return samples;
        std::shuffle(samples.begin(), samples.end(), rng);
        return vector<int>(samples.begin(), samples.begin() + num);
    }
    
    vector<int> generateFeatures(function<int(int)> func) const {
        int m = func(getFeatureSize());
        vector<int> allFeatures;
        for (int i = 0; i < getFeatureSize(); i++) {
            allFeatures.push_back(i);
        }
        std::shuffle(allFeatures.begin(), allFeatures.end(), rng);
        if(m > (int)allFeatures.size()) m = allFeatures.size();
        return vector<int>(allFeatures.begin(), allFeatures.begin() + m);
    }
    
    void sortByFeature(vector<int> &indices, int featureIndex) const {
        sort(indices.begin(), indices.end(), [this, featureIndex](int a, int b){
            return readFeature(a, featureIndex) < readFeature(b, featureIndex);
        });
    }
    
    const vector<int>& getTarget() const {
        return target;
    }
};

// ---------------- Decision Tree Functions ----------------

int computeTrueCount(const vector<int>& samples, const Data &data) {
    int total = 0;
    for(auto idx : samples)
        total += data.readTarget(idx);
    return total;
}

double computeTargetProb(const vector<int>& samples, const Data &data) {
    double sum = 0;
    int count = 0;
    for(auto idx : samples) {
        sum += data.readTarget(idx);
        count++;
    }
    return sum / (count + 1e-9);
}

double computeGini(int trueCount, int groupSize) {
    double p = trueCount / (double)(groupSize + 1e-9);
    return 1 - p * p - (1 - p) * (1 - p);
}

double computeGiniIndex(int leftTrue, int leftSize, int rightTrue, int rightSize) {
    int total = leftSize + rightSize;
    double leftWeight = leftSize / (double) total;
    double rightWeight = rightSize / (double) total;
    return leftWeight * computeGini(leftTrue, leftSize) + rightWeight * computeGini(rightTrue, rightSize);
}

// ---------------- Helper Functions for Feature Selection ----------------

int _sqrt(int num) {
    return max(1, (int) sqrt(num));
}

int _log2(int num) {
    return max(1, (int) log2(num));
}

int _none(int num) {
    return num;
}

// ---------------- DecisionTree Class ----------------

class DecisionTree {
private:
    struct Node {
        int featureIndex;
        double threshold;
        bool isLeaf;
        double prob;
        shared_ptr<Node> left;
        shared_ptr<Node> right;
        int depth;
        Node() : featureIndex(-1), threshold(0), isLeaf(false), prob(0), depth(0) {}
    };
    
    shared_ptr<Node> root;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    int sampleNum;
    function<double(int, int, int, int)> criterionFunc;
    function<int(int)> maxFeatureFunc;
    
    shared_ptr<Node> constructNode(const vector<int>& samples, const Data &data, int depth) {
        double prob = computeTargetProb(samples, data);
        auto node = make_shared<Node>();
        node->depth = depth;
        if(prob == 0.0 || prob == 1.0 || samples.size() <= (size_t)minSamplesSplit ||
           (maxDepth != -1 && depth >= maxDepth)) {
            node->isLeaf = true;
            node->prob = prob;
            return node;
        }
        vector<int> features = data.generateFeatures(maxFeatureFunc);
        double bestGini = 1e9;
        int bestFeature = features[0];
        double bestThreshold = 0;
        for (int feat : features) {
            set<double> values;
            for (int idx : samples)
                values.insert(data.readFeature(idx, feat));
            for (auto val : values) {
                vector<int> left, right;
                int leftTrue = 0, rightTrue = 0;
                for (int idx : samples) {
                    if(data.readFeature(idx, feat) <= val) {
                        left.push_back(idx);
                        leftTrue += data.readTarget(idx);
                    } else {
                        right.push_back(idx);
                        rightTrue += data.readTarget(idx);
                    }
                }
                if(left.size() < (size_t)minSamplesLeaf || right.size() < (size_t)minSamplesLeaf)
                    continue;
                double gini = computeGiniIndex(leftTrue, left.size(), rightTrue, right.size());
                if(gini < bestGini) {
                    bestGini = gini;
                    bestFeature = feat;
                    bestThreshold = val;
                }
            }
        }
        vector<int> left, right;
        for (int idx : samples) {
            if(data.readFeature(idx, bestFeature) <= bestThreshold)
                left.push_back(idx);
            else
                right.push_back(idx);
        }
        if(left.size() < (size_t)minSamplesLeaf || right.size() < (size_t)minSamplesLeaf) {
            node->isLeaf = true;
            node->prob = prob;
            return node;
        }
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = constructNode(left, data, depth + 1);
        node->right = constructNode(right, data, depth + 1);
        return node;
    }
    
public:
    DecisionTree(const string &criterion = "gini", int maxDepth = -1, int minSamplesSplit = 2,
                 int minSamplesLeaf = 1, int sampleNum = -1, const string &maxFeatures = "auto")
      : maxDepth(maxDepth), minSamplesSplit(minSamplesSplit),
        minSamplesLeaf(minSamplesLeaf), sampleNum(sampleNum) {
        if(criterion == "gini")
            criterionFunc = computeGiniIndex;
        else
            criterionFunc = computeGiniIndex; // default to gini
        if(maxFeatures == "auto" || maxFeatures == "sqrt")
            maxFeatureFunc = _sqrt;
        else if(maxFeatures == "log2")
            maxFeatureFunc = _log2;
        else
            maxFeatureFunc = _none;
    }
    
    void fit(const Data &data) {
        vector<int> samples = data.generateSample(sampleNum);
        root = constructNode(samples, data, 0);
    }
    
    double computeProb(int sampleIndex, const Data &data) const {
        auto node = root;
        while(!node->isLeaf) {
            if(data.readFeature(sampleIndex, node->featureIndex) <= node->threshold)
                node = node->left;
            else
                node = node->right;
        }
        return node->prob;
    }
    
    void predictProba(const Data &data, vector<double> &results) const {
        int n = data.getSampleSize();
        for (int i = 0; i < n; i++) {
            results[i] += computeProb(i, data);
        }
    }
};

// ---------------- RandomForest Class (OpenMP) ----------------

class RandomForest {
private:
    vector<DecisionTree> trees;
    int nEstimators;
public:
    RandomForest(int nEstimators = 10, string criterion = "gini", string maxFeatures = "auto",
                 int maxDepth = -1, int minSamplesSplit = 2, int minSamplesLeaf = 1,
                 int eachTreeSamplesNum = -1)
      : nEstimators(nEstimators) {
        trees.reserve(nEstimators);
        for (int i = 0; i < nEstimators; i++) {
            trees.push_back(DecisionTree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                                           eachTreeSamplesNum, maxFeatures));
        }
    }
    
    // Parallel tree fitting using OpenMP.
    void fit(const Data &data) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nEstimators; i++) {
            trees[i].fit(data);
            #pragma omp critical
            {
                //cout << "Fitted tree " << i + 1 << "/" << nEstimators << endl;
            }
        }
    }
    
    // Parallel prediction: each tree computes predictions in parallel.
    vector<double> predictProba(const Data &data) {
        int n = data.getSampleSize();
        vector<vector<double>> allPred(nEstimators, vector<double>(n, 0.0));
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nEstimators; i++) {
            trees[i].predictProba(data, allPred[i]);
            #pragma omp critical
            {
                //cout << "Predicted with tree " << i + 1 << "/" << nEstimators << endl;
            }
        }
        vector<double> results(n, 0.0);
        for (int i = 0; i < nEstimators; i++) {
            for (int j = 0; j < n; j++) {
                results[j] += allPred[i][j];
            }
        }
        for (int i = 0; i < n; i++)
            results[i] /= nEstimators;
        return results;
    }
};

// ---------------- Main ----------------

int main() {
    auto start_total = high_resolution_clock::now();
    
    int trainSize = 30000;
    int testSize  = 1000;
    int featureSize = 107;
    
    Data trainData(true, trainSize, featureSize);
    trainData.read("train.txt");
    
    RandomForest rf(2000, "gini", "log2", 3, 150, 1, 1000000);
    auto start_fit = high_resolution_clock::now();
    rf.fit(trainData);
    auto end_fit = high_resolution_clock::now();
    auto fit_duration = duration_cast<milliseconds>(end_fit - start_fit);
    
    vector<double> trainPred = rf.predictProba(trainData);
    double aucTrain = computeAUC(trainPred, trainData.getTarget());
    double accTrain = computeAccuracy(trainPred, trainData.getTarget());
    
    Data testData(true, testSize, featureSize);
    testData.read("test.txt");
    
    auto start_pred = high_resolution_clock::now();
    vector<double> testPred = rf.predictProba(testData);
    auto end_pred = high_resolution_clock::now();
    auto pred_duration = duration_cast<milliseconds>(end_pred - start_pred);
    
    double aucTest = computeAUC(testPred, testData.getTarget());
    double accTest = computeAccuracy(testPred, testData.getTarget());
    
    writeDataToCSV(testPred, testData.getTarget(), "results_openmp.csv", true);
    
    auto end_total = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_total - start_total);
    
    cout << "Training time (OpenMP): " << fit_duration.count() << " ms" << endl;
    cout << "Prediction time (OpenMP): " << pred_duration.count() << " ms" << endl;
    cout << "Total time (OpenMP): " << total_duration.count() << " ms" << endl;
    cout << "  Train Accuracy: " << accTrain << endl;
    cout << "  Test Accuracy: "  << accTest  << endl;
    
    return 0;
}

#ifndef NARX_DARNN_H
#define NARX_DARNN_H

#include <memory>
#include <functional>

class NARX_DARNN
{
public:
    NARX_DARNN(int inDim, int delay, int stride);
    virtual ~NARX_DARNN();

    bool loadData(const std::string &filename,
                  const int inDim,
                  const int startRow = 0);

    /**
     * @brief train
     * @param batch
     * @param nEpoches
     * @param learningRate
     * @param cb
     * @param validateEpoch if set to > 0, a validation test over with given data will be conducted every validateEpoch
     * @param cb2 Being the callback when validateEpoch > 0
     * @return
     */
    bool train(const int batch = 4,
               const int nEpoches = 100,
               const double learningRate = 1e-3,
               std::function<void (int, double)> *cb = nullptr,
               const int validateEpoch = -1,
               std::function<void (int, double, double)> *cb2 = nullptr);

    /**
     * @brief Call loadData() to append the history data before calling test().
     * @param file
     * @param cb
     * @return
     */
    bool test( const std::string &file,
               std::function<void (int, double, double)> *cb = nullptr);

    static void init();
private:
    struct member;
    std::shared_ptr<member> m;
};



#endif // NARX_DARNN_H

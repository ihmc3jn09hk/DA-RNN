#include "narx_darnn.h"
#include <torch/torch.h>

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int64_t inDim,
                int64_t delay,
                int64_t layerWidth)
        : m_inDim(inDim),
          m_delay(delay),
          m_decWidth(layerWidth)
    {
        linear = register_module("Linear", torch::nn::Linear(m_inDim+1, 1));
        linear->weight.data().normal_();

        linear_out = register_module("Linear_Out", torch::nn::Linear(m_inDim+m_decWidth, 1));

        attd = register_module("Attd", torch::nn::Sequential(
                    torch::nn::Linear(2*m_decWidth + m_inDim, m_inDim ),
                    torch::nn::Tanh(),
                    torch::nn::Linear(m_inDim, 1)
                    ));
        auto lstm_options = torch::nn::LSTMOptions(1, m_decWidth);
        lstm_options.num_layers(1);
        lstm = register_module("LSTM",torch::nn::LSTM(lstm_options));

    }
    torch::Tensor forward(torch::Tensor X_enc, torch::Tensor y_delay) {
        auto options = torch::TensorOptions();
        options = options.requires_grad(true);
        auto dN = X_enc.new_zeros(c10::IntArrayRef({1, X_enc.size(0), m_decWidth}), options);
        auto cN = X_enc.new_zeros(c10::IntArrayRef({1, X_enc.size(0), m_decWidth}), options);

        const auto py_colon = at::indexing::Slice(at::indexing::None, at::indexing::None);

        torch::Tensor context;
        for ( int t=0; t<m_delay; ++t ) {
            auto x = torch::cat({dN.repeat({m_delay, 1, 1}).permute({1, 0, 2}),
                                 cN.repeat({m_delay, 1, 1}).permute({1, 0, 2}),
                                X_enc}, 2);
            auto _b = torch::nn::functional::softmax(
                        attd->forward(x.view({-1, 2*m_decWidth + m_inDim})).view({-1, m_delay}), 1
                        );

            context = torch::bmm( _b.unsqueeze(1), X_enc).index(
                        {py_colon, 0, py_colon});
            if ( t < m_delay-1 ) {
                auto y_tilde = linear->forward(
                            torch::cat({context, y_delay.index({py_colon, t}).unsqueeze(1)}, 1));

                lstm->flatten_parameters();
                auto [rc, outStatues] = lstm->forward(y_tilde.unsqueeze(0),
                        std::tuple<at::Tensor, at::Tensor>{dN, cN});

                dN = std::get<0>(outStatues);
                cN = std::get<1>(outStatues);
            }
        }
        auto y_pred = linear_out->forward(torch::cat({dN[0], context}, 1));
        return y_pred;
    }

    int64_t m_inDim;
    int64_t m_delay;
    int64_t m_decWidth;
    torch::nn::Linear linear = nullptr,
                      linear_out = nullptr;
    torch::nn::Sequential attd = nullptr;
    torch::nn::LSTM lstm = nullptr;
};
TORCH_MODULE(Decoder);

struct EncoderImpl : torch::nn::Module {
    EncoderImpl(int64_t inDim,
                int64_t delay,
                int64_t layerWidth):
        m_inDim(inDim),
        m_delay(delay),
        m_encWidth(layerWidth)
    {
        m_attd = register_module("Attd", torch::nn::Linear(
                                     2 * m_encWidth + delay,    //in_feature
                                     1                          //out_feature
                                     ));
        m_lstm = register_module("LSTM", torch::nn::LSTM(m_inDim,   //input_size
                                                         m_encWidth //hidden_size
                                                         ));
    }
    std::tuple<at::Tensor, at::Tensor> forward(torch::Tensor X) {
        auto options = torch::TensorOptions();
        options = options.requires_grad(true);
        auto X_tilde = X.new_zeros({X.size(0), m_delay, m_inDim}, options);
        auto X_enc = X.new_zeros({X.size(0), m_delay, m_encWidth}, options);

        auto hN = X.new_zeros({1, X.size(0), m_encWidth}, options);
        auto sN = X.new_zeros({1, X.size(0), m_encWidth}, options);

        const auto py_colon = at::indexing::Slice(at::indexing::None, at::indexing::None);

        for (int t=0; t<m_delay; ++t ) {
            auto x = torch::cat({hN.repeat({m_inDim, 1, 1}).permute({1, 0, 2}),
                                 sN.repeat({m_inDim, 1, 1}).permute({1, 0, 2}),
                                 X.permute({0, 2, 1})},
                                2 );
            x = m_attd->forward(x.view({-1, 2 * m_encWidth + m_delay}));

            auto a_ = torch::nn::functional::softmax(x.view({-1, m_inDim}),1);
            auto x_tilde = torch::mul(a_, X.index({py_colon, t, py_colon}));

            m_lstm->flatten_parameters();
            auto [rc, encState] = m_lstm->forward(x_tilde.unsqueeze(0), std::tuple<at::Tensor, at::Tensor>{hN, sN});

            hN = std::get<0>(encState);
            sN = std::get<1>(encState);

            X_tilde.index({py_colon, t, py_colon}) = x_tilde;
            X_enc.index({py_colon, t, py_colon}) = hN[0];

        }
      return std::make_tuple(X_tilde, X_enc);
    }
    int64_t m_inDim;
    int64_t m_delay;
    int64_t m_encWidth;
    torch::nn::Linear m_attd = nullptr;
    torch::nn::LSTM m_lstm = nullptr;
};
TORCH_MODULE(Encoder);

struct DARNNImpl : torch::nn::Module {
    DARNNImpl(  int64_t inDim,
                int64_t delay,
                int64_t encLayerWidth = 128,
                int64_t decLayerWidth = 128):
        m_inDim(inDim),
        m_delay(delay),
        m_encWidth(encLayerWidth),
        m_decWidth(decLayerWidth)
    {
        encoder = register_module("Encoder", Encoder(inDim, delay, encLayerWidth));
        decoder = register_module("Decoder", Decoder(encLayerWidth, delay, decLayerWidth));
    }

    torch::Tensor forward(torch::Tensor X, torch::Tensor y_prev) {
        auto [input_weighted, input_encoded] = encoder->forward(X);
        auto y_pred = decoder->forward(input_encoded, y_prev);

        return y_pred;
    }

    torch::nn::Linear linear = nullptr;

    int64_t m_inDim;
    int64_t m_delay;
    int64_t m_encWidth;
    int64_t m_decWidth;

    Encoder encoder = nullptr;
    Decoder decoder = nullptr;

};
TORCH_MODULE(DARNN);

struct NARX_DARNN::member {
    DARNN net = nullptr;

    int inDim = 4;
    int delay = 4;
    int stride = 672;

    std::shared_ptr<torch::Device> device;
    torch::Tensor X_train, X_validate;
    torch::Tensor y_train, y_validate;
    torch::Tensor readcsv(const std::string &file, torch::Device device,
                          int startRow = 0, int endRow = -1, int startCol = 0) const;
};

NARX_DARNN::NARX_DARNN(int inDim, int delay, int stride) :
    m(std::make_shared<member>())
{
    m->device = std::make_shared<torch::Device>(torch::cuda::is_available() ? "cuda" : "cpu");
    m->inDim = inDim;
    m->delay = delay;
    m->stride = stride;

    m->net = DARNN(4, m->delay);
    torch::load(m->net, "model_5000.pt");
    m->net->to(*m->device);

    std::cout << m->net << std::endl;
}

NARX_DARNN::~NARX_DARNN()
{
    m.reset();
}

bool NARX_DARNN::loadData(const std::string &filename, const int inDim, const int startRow)
{
    auto towerID = 4;   //"4 == NT", 5 == ST
    auto X = m->readcsv(filename, *m->device, startRow, -1);   //Skip the first row
    if ( 0 == X.size(0)) {
        return false;
    }

    const auto py_colon = at::indexing::Slice(at::indexing::None, at::indexing::None);

    m->X_train = X.index({py_colon, at::indexing::Slice(0, inDim)});
    m->y_train = X.index({py_colon, towerID});

    return true;
}

bool NARX_DARNN::train(const int batch, const int nEpoches,
                       const double learningRate,
                       std::function<void (int, double)> *cb)
{
    printf("Prepare Training");
    const int nRows = m->X_train.size(0);
    auto criterion = std::bind(torch::mse_loss, std::placeholders::_1, std::placeholders::_2, at::Reduction::Sum);

    //Optimizer
    auto options = torch::optim::AdamOptions(learningRate);
    auto encOptim = std::make_shared<torch::optim::Adam>(m->net->encoder->parameters(), options);
    auto decOptim = std::make_shared<torch::optim::Adam>(m->net->decoder->parameters(), options);

    m->net->train();

    //Prepare Samples
    torch::TensorOptions tensorOptions;
    tensorOptions = tensorOptions.device(*m->device);
    std::vector<torch::Tensor> X_train, Y_delay, Y_label;
    bool bContinue = true;
    int j = nRows;
    const int strideXdelay = m->stride*m->delay;
    const int strideXdelayM1 = m->stride*(m->delay-1);

    std::string msg;
    msg = std::to_string(j) + " / " + std::to_string(nRows);

    std::cout << "Remain to be converted : " << msg;

    while (bContinue) {
        int startIdx = j-batch - m->stride*(m->delay-1);
        if ( 0 > startIdx ) {
            j = batch + m->stride*(m->delay-1);
            startIdx = 0;
            bContinue = false;
        }

        auto x = torch::zeros({batch, m->delay, m->inDim}, tensorOptions);
        auto y_delay = torch::zeros({batch, m->delay - 1}, tensorOptions);
        auto y_label = m->y_train.index({at::indexing::Slice(j-batch, j)});

        for ( int k=0; k<batch; ++k ) {
            const auto rowId = startIdx+k;
            y_delay[k] = m->y_train.index({at::indexing::Slice(
                                           rowId,
                                           rowId + strideXdelayM1,
                                           m->stride)});
            x[k] = m->X_train.index({at::indexing::Slice(
                                           rowId,
                                           rowId + strideXdelay,
                                           m->stride)});
        }
        y_label = y_label.view({-1,1});

        X_train.push_back(x);
        Y_delay.push_back(y_delay);
        Y_label.push_back(y_label);

        j -= batch;

        msg = std::to_string(j) + " / " + std::to_string(nRows);

        std::cout << "\rRemain to be converted : "
                  << msg
                  << std::flush;
    }
    std::cout << std::endl;

    //
    printf("Start training. Batch : %u, Epoches : %u\n", batch, nEpoches);

    for ( int i=0; i<nEpoches; ++i ) {
        double eloss = 0;

        const int nBatches = X_train.size();

        msg = "0%";
        std::cout << "Iteration : " << msg;

        for ( j=0; j<nBatches; ++j ) {

            encOptim->zero_grad();
            decOptim->zero_grad();

            auto &x = X_train.at(j);
            auto &y_delay = Y_delay.at(j);
            auto &y_label = Y_label.at(j);
            auto y_pred = m->net->forward(x, y_delay);

            auto loss = criterion(y_label, y_pred);
            loss.backward();

            encOptim->step();
            decOptim->step();

            eloss += loss.item().toDouble();

            msg = std::to_string(double(100*j*batch) / nRows) + "%" ;
            std::cout << "\rIteration : " << msg << std::flush;
        }
        std::cout << std::endl;
        eloss /= nRows;

        std::cout << "Epoch " << i << "/" << nEpoches
                  << " finished with loss : " << eloss
                  << std::endl;
        if ( cb ) {
            (*cb)(i, eloss);
        }
        if ( 0 == (i+1) % 100 ) {
            char filename[32];
            sprintf(filename, "model_%d.pt", i+1);
            std::cout << "Save model " << filename << std::endl;
            torch::save(m->net, filename);
        }
    }
    return true;
}

bool NARX_DARNN::test(const std::string &file,
                      std::function<void (int, double, double)> *cb)
{
    auto _X = m->readcsv(file, *m->device, 1, -1, 1);
    if ( 0 == _X.size(0)) {
        return false;
    }

    auto towerID = 4;   //"4 == NT", 5 == ST
    const auto py_colon = at::indexing::Slice(at::indexing::None, at::indexing::None);

    auto X = _X.index({py_colon, at::indexing::Slice(0, m->inDim)});
    bool bHasValue = false;
    auto _Y = torch::Tensor();
    if ( m->inDim < _X.size(1)) {
        bHasValue = true;
        _Y = _X.index({py_colon, towerID});
    }

    auto Y = torch::zeros({_X.size(0)}, X.options());

    X = torch::cat({m->X_train, X}, 0).to(*m->device);
    Y = torch::cat({m->y_train, Y}, 0).to(*m->device);

    const int nRows = X.size(0);
    const int historyEnd = m->X_train.size(0);

    //Prepare Samples
    const int strideXdelay = m->stride*m->delay;
    const int strideXdelayM1 = m->stride*(m->delay-1);

    m->net->eval();
    torch::NoGradGuard no_grad;

    int lastProgress = 0;
    const int nTestSample = nRows - historyEnd;
    std::string msg = "[ 0.0% ]\r";
    std::cout << msg << std::flush;

    for ( int i=historyEnd; i<nRows; ++i ) {
        int startIdx = i - m->stride*(m->delay-1);
        const auto rowId = startIdx;

        auto y_delay = Y.index({at::indexing::Slice(
                                       rowId,
                                       rowId + strideXdelayM1,
                                       m->stride)});
        auto x = X.index({at::indexing::Slice(
                                       rowId,
                                       rowId + strideXdelay,
                                       m->stride)});

        x = x.unsqueeze(0);
        y_delay = y_delay.unsqueeze(0);

        auto y_pred = m->net->forward(x, y_delay).flatten();
        Y[i] = y_pred.item();

        const float curProgress = 100.f*(i-historyEnd+1) / nTestSample;
        if ( lastProgress != int(curProgress)) {
            msg = "[";
            for ( int p=0; p<int(0.55*curProgress)-1; ++p ) {
                msg += "-";
            }
            msg += ">";
            for ( int p=int(0.55*curProgress); p<55; ++p ) {
                msg += " ";
            }
            msg += "]";

            char percentage[9];
            sprintf(percentage, " %4.1f%% ", curProgress);

            std::cout << msg
                      << std::string(32, '\b')
                      << std::string(percentage);
            std::cout << '\r' << std::flush;

            lastProgress = int(curProgress);
        }

        if ( cb ) {
            double y_true = 0.0;
            if ( bHasValue ) {
                y_true = _Y[i - historyEnd].item().toDouble();
            }
            (*cb)(i - historyEnd, y_pred.item().toDouble(), y_true);
        }
    }
    std::cout << std::endl;

    return true;
}

void NARX_DARNN::init()
{
    torch::init();
}

torch::Tensor NARX_DARNN::member::readcsv(const std::string &file,
                                          torch::Device device,
                                          int startRow,
                                          int endRow,
                                          int startCol) const
{
    std::ifstream in(file, std::ios::in);
    if (!in.is_open()){
        return torch::zeros({0});
    }

    // Read the Data from the file
    // as String Vector
    std::vector<std::string> col;
    std::vector<std::vector<std::string>> rows;
    std::string line, word, temp;

    for ( int i=0; i<startRow; ++i ) {
        std::getline(in, line);
    }

    while (std::getline(in, line)) {
        col.clear();
        if ( line.empty()) {
            continue;
        }
        // used for breaking words
        std::stringstream s(line);

        // read every column data of a row and
        // store it in a string variable, 'word'
        int cur_col = 0;
        while (std::getline(s, word, ',')) {
            if ( startCol > cur_col++) {
                continue;
            }
            // add all the column data
            // of a row to a vector
            col.push_back(word);
        }

        rows.push_back(col);
        if ( endRow == int(rows.size())) {
            break;
        }
    }
    in.close();

    const int nRows = rows.size();
    if ( 0 == nRows ) {
        std::cout << "Invalid Inputs" << std::endl;
        return torch::zeros({0});
    }
    //Assume every row has identical number of entries
    const int nCols = rows.front().size();

    torch::TensorOptions tensorOptions;
    tensorOptions = tensorOptions.device(device);

    auto X = torch::zeros({nRows, nCols}, tensorOptions);
    for ( int i=0; i<nRows; ++i ) {
        const auto &row = rows.at(i);
        for ( int j=0; j<nCols; ++j ) {
            X.index({i,j}) = std::stod(row.at(j));
        }
    }
    return X;
}

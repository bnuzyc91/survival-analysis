/*!
 * Copyright 2015 by Contributors
 * \file elementwise_metric.cc
 * \brief evaluation metrics for elementwise binary or regression.
 * \author Kailong Chen, Tianqi Chen
 */
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>
#include "../common/math.h"
#include "../common/sync.h"

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(elementwise_metric);

/*!
 * \brief base class of element-wise evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalEWiseBase : public Metric {
  bst_float Eval(const std::vector<bst_float>& preds,
                 const MetaInfo& info,
                 bool distributed) const override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    const omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
    double sum = 0.0, wsum = 0.0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const bst_float wt = info.GetWeight(i);
      sum += static_cast<const Derived*>(this)->EvalRow(info.labels[i], preds[i]) * wt;
      wsum += wt;
    }
    double dat[2]; dat[0] = sum, dat[1] = wsum;
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Derived::GetFinal(dat[0], dat[1]);
  }
  /*!
   * \brief to be implemented by subclass,
   *   get evaluation result from one row
   * \param label label of current instance
   * \param pred prediction value of current instance
   */
  inline bst_float EvalRow(bst_float label, bst_float pred) const;
  /*!
   * \brief to be overridden by subclass, final transformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

struct EvalRMSE : public EvalEWiseBase<EvalRMSE> {
  const char *Name() const override {
    return "rmse";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return std::sqrt(esum / wsum);
  }
};

struct EvalMAE : public EvalEWiseBase<EvalMAE> {
  const char *Name() const override {
    return "mae";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    return std::abs(label - pred);
  }
};

struct EvalLogLoss : public EvalEWiseBase<EvalLogLoss> {
  const char *Name() const override {
    return "logloss";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    const bst_float pneg = 1.0f - py;
    if (py < eps) {
      return -y * std::log(eps) - (1.0f - y)  * std::log(1.0f - eps);
    } else if (pneg < eps) {
      return -y * std::log(1.0f - eps) - (1.0f - y)  * std::log(eps);
    } else {
      return -y * std::log(py) - (1.0f - y) * std::log(pneg);
    }
  }
};

struct EvalError : public EvalEWiseBase<EvalError> {
  explicit EvalError(const char* param) {
    if (param != nullptr) {
      std::ostringstream os;
      os << "error";
      CHECK_EQ(sscanf(param, "%f", &threshold_), 1)
        << "unable to parse the threshold value for the error metric";
      if (threshold_ != 0.5f) os << '@' << threshold_;
      name_ = os.str();
    } else {
      threshold_ = 0.5f;
      name_ = "error";
    }
  }
  const char *Name() const override {
    return name_.c_str();
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }
 protected:
  bst_float threshold_;
  std::string name_;
};

struct EvalPoissonNegLogLik : public EvalEWiseBase<EvalPoissonNegLogLik> {
  const char *Name() const override {
    return "poisson-nloglik";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    if (py < eps) py = eps;
    return common::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }
};

struct EvalGammaDeviance : public EvalEWiseBase<EvalGammaDeviance> {
  const char *Name() const override {
    return "gamma-deviance";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float epsilon = 1.0e-9;
    bst_float tmp = label / (pred + epsilon);
    return tmp - std::log(tmp) - 1;
  }
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return 2 * esum;
  }
};

struct EvalGammaNLogLik: public EvalEWiseBase<EvalGammaNLogLik> {
  const char *Name() const override {
    return "gamma-nloglik";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
    bst_float psi = 1.0;
    bst_float theta = -1. / py;
    bst_float a = psi;
    bst_float b = -std::log(-theta);
    bst_float c = 1. / psi * std::log(y/psi) - std::log(y) - common::LogGamma(1. / psi);
    return -((y * theta - b) / a + c);
  }
};

struct EvalTweedieNLogLik: public EvalEWiseBase<EvalTweedieNLogLik> {
  explicit EvalTweedieNLogLik(const char* param) {
    CHECK(param != nullptr)
        << "tweedie-nloglik must be in format tweedie-nloglik@rho";
    rho_ = atof(param);
    CHECK(rho_ < 2 && rho_ >= 1)
        << "tweedie variance power must be in interval [1, 2)";
    std::ostringstream os;
    os << "tweedie-nloglik@" << rho_;
    name_ = os.str();
  }
  const char *Name() const override {
    return name_.c_str();
  }
  inline bst_float EvalRow(bst_float y, bst_float p) const {
    bst_float a = y * std::exp((1 - rho_) * std::log(p)) / (1 - rho_);
    bst_float b = std::exp((2 - rho_) * std::log(p)) / (2 - rho_);
    return -a + b;
  }
 protected:
  std::string name_;
  bst_float rho_;
};

        void calc_dn(
                const std::vector<bool> &conversion_event,
                const std::vector<bst_ulong> &time_to_convert,
                bst_ulong max_time,
                std::vector<bst_ulong> *out_dn
        ){
            for (size_t i=0;i <time_to_convert.size(); ++i){
                auto event = conversion_event[i];
                auto time = time_to_convert[i];
                if (event){
                    out_dn->at(time)+=1;
                }
            }
        }

  void calc_second_part_denom(
          const std::vector<double> &exp_preds,
          bst_ulong max_time,
          const std::vector<bst_ulong> &time_to_convert,
          std::vector<double> *out_second_part_denom
  )
  {
     for (size_t t=0; t< out_second_part_denom->size();++t)
      {
          double out_second_part_denom_t = 0.0f;
          for (size_t i=0; i<time_to_convert.size(); ++i){
              if (time_to_convert[i] >= t){
                  out_second_part_denom_t += exp_preds[i];
              }
          }
          out_second_part_denom->at(t) = out_second_part_denom_t;
      }
  }


        void calc_second_part_denom_optimized(
                const std::vector<double> &exp_preds,
                bst_ulong max_time,
                const std::vector<bst_ulong> &time_to_convert,
                std::vector<double> *out_second_part_denom) {
            auto prev_time = max_time;
            double sum_y = 0.0f;
            for (int i = time_to_convert.size() - 1; i >= 0; --i) {
                auto time = time_to_convert[i];
                if (prev_time != time) {
                    for (int t = prev_time; t > time; --t) {
                        (*out_second_part_denom)[t] = sum_y;
                    }
                }
                sum_y += exp_preds[i];
                prev_time = time;
            }
            for (int t = prev_time; t >= 0; --t) {
                (*out_second_part_denom)[t] = sum_y;
            }
        }
        double get_log_likelihood(
                const std::vector<bool> &conversion_event,
                const std::vector<double> &second_part_denom,
                const std::vector<bst_ulong> &d,
                const std::vector<double > &preds, //here double
                bst_ulong max_time
        ){
            double second_part = 0.0f;
            for (size_t n=0; n < d.size(); ++n)
            {
                if (d[n]==0){
                    continue;
                }
                //second_part+=d[n] * std::log(second_part_denom[n]);
                //change here
                second_part+=d[n] * std::log(second_part_denom[n]);


            }
            double first_part = 0.0f;
            for (size_t i=0;i <preds.size(); ++i){
                if (conversion_event[i]){
                    first_part += std::log(preds[i]+0.00001);
                     //change here
                    if(std::isinf(first_part))
                     std::cout << "i pred is"  << preds[i] << "\n";
                   // first_part += std::abs(preds[i]);

                }
            }
             std::cout << "first_part = "  << first_part << "\n";
             std::cout << "second_part = "  << second_part << "\n";
            return second_part - first_part;
        }
        struct EvalCIndex : public Metric {

            bst_float Eval(const std::vector<bst_float> &preds,
                           const MetaInfo &info,
                           bool distributed) const override {
                CHECK(!distributed) << "metric concordance_index do not support distributed evaluation";

                std::vector<bst_ulong> time_to_convert(info.labels.begin(), info.labels.end());
                std::vector<bool> conversion_event(info.censor.begin(), info.censor.end());
                double concordance=0.0f;
                double permissible=0.0f;
                for (size_t i = 0; i < preds.size()-1; ++i) {
                    for (size_t j = i+1; j < preds.size(); ++j) {
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==0) && (conversion_event[j]==0)){
                          continue;
                          }
                        if((time_to_convert[i]<time_to_convert[j]) && (conversion_event[i]==0) ){
                          continue;
                          }
                        if((time_to_convert[j]<time_to_convert[i]) && (conversion_event[j]==0) ){
                          continue;
                          }

                        permissible=permissible+1;

                        if((time_to_convert[i]<time_to_convert[j]) && (preds[i]<preds[j])) {
                          concordance=concordance+1;
                          }
                        if((time_to_convert[i]>time_to_convert[j]) && (preds[i]>preds[j])){
                          concordance=concordance+1;
                          }
                        if((time_to_convert[i]!=time_to_convert[j]) && (preds[i]==preds[j])) {
                          concordance=concordance+0.5;
                          }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==1) && (conversion_event[j]==1) && (preds[i]==preds[j])){
                          concordance=concordance+1;
                          }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==1) && (conversion_event[j]==1) && (preds[i]!=preds[j]))
                        {
                          concordance=concordance+0.5;
                        }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==1) && (conversion_event[j]==0) &&  (preds[i]<preds[j]))
                        {
                          concordance=concordance+1;
                        }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==0) && (conversion_event[j]==1) && (preds[i]>preds[j]))
                        {
                          concordance=concordance+1;
                        }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==1) && (conversion_event[j]==0) && (preds[i]>=preds[j]))
                        {
                          concordance=concordance+0.5;
                        }
                        if((time_to_convert[i]==time_to_convert[j]) && (conversion_event[i]==0) && (conversion_event[j]==1) && (preds[i]<=preds[j])) {
                          concordance=concordance+0.5;
                        }
                    }
                }
                 auto CIndex=concordance / permissible;
                // std::cout << "concordance = "  << concordance << "\n";
                 //std::cout << "permissible = "  << permissible << "\n";
                return 1.0-CIndex;
            }
            const char *Name() const override {
                return "CIndex";
            }
        };

    struct EvalPartialLikelihood : public Metric {

            bst_float Eval(const std::vector<bst_float> &preds,
                           const MetaInfo &info,
                           bool distributed) const override {
                CHECK(!distributed) << "metric partial_likelihood do not support distributed evaluation";
                std::vector<double> preds_center(preds.size(),0.0f);
                std::vector<double> exp_preds(preds.size(),0.0f);
                double maxpred=0.0f;
                for (size_t i = 0; i < preds.size(); ++i) {
                    if (preds[i] > maxpred ) maxpred=preds[i];
                }
                std::cout << "maxpred = "  << maxpred << "\n";
                //center based on the max pred change here
                for (size_t i = 0; i < preds.size(); ++i) {
                    preds_center[i]= preds[i]-maxpred;
                }
                maxpred=0.0;
                for (size_t i = 0; i < exp_preds.size(); ++i) {
                    double tmp=(double)  preds_center[i];
                    exp_preds[i] = std::exp(tmp);
                    if (exp_preds[i] > maxpred ) maxpred=exp_preds[i];
                }

                std::cout << "max expential pred = "  << maxpred << "\n";
                std::vector<bst_ulong> time_to_convert(info.labels.begin(), info.labels.end());
                std::vector<bool> conversion_event(info.censor.begin(), info.censor.end());
                auto max_time = *std::max_element(time_to_convert.begin(), time_to_convert.end());
                std::vector<bst_ulong> d_n(max_time+1, 0);
                std::vector<double> second_part_denom(max_time+1, 0.0f);
                std::vector<double> second_part_denom_opt(max_time+1, 0.0f);

                std::vector<double> second_part(max_time+1, 0.0f);
                calc_dn(conversion_event, time_to_convert, max_time, &d_n);
                //calc_second_part_denom_optimized(exp_preds, max_time, time_to_convert, &second_part_denom);
                calc_second_part_denom(exp_preds, max_time, time_to_convert, &second_part_denom);

                auto partial_likelihood = get_log_likelihood(conversion_event, second_part_denom, d_n, exp_preds, max_time);
                //change here
               // auto partial_likelihood = get_log_likelihood(conversion_event, second_part_denom, d_n, preds, max_time);
                return partial_likelihood;
            }
            const char *Name() const override {
                return "partial_likelihood";
            }
        };

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
.describe("Rooted mean square error.")
.set_body([](const char* param) { return new EvalRMSE(); });

XGBOOST_REGISTER_METRIC(MAE, "mae")
.describe("Mean absolute error.")
.set_body([](const char* param) { return new EvalMAE(); });

XGBOOST_REGISTER_METRIC(LogLoss, "logloss")
.describe("Negative loglikelihood for logistic regression.")
.set_body([](const char* param) { return new EvalLogLoss(); });

XGBOOST_REGISTER_METRIC(Error, "error")
.describe("Binary classification error.")
.set_body([](const char* param) { return new EvalError(param); });

XGBOOST_REGISTER_METRIC(PossionNegLoglik, "poisson-nloglik")
.describe("Negative loglikelihood for poisson regression.")
.set_body([](const char* param) { return new EvalPoissonNegLogLik(); });

XGBOOST_REGISTER_METRIC(GammaDeviance, "gamma-deviance")
.describe("Residual deviance for gamma regression.")
.set_body([](const char* param) { return new EvalGammaDeviance(); });

XGBOOST_REGISTER_METRIC(GammaNLogLik, "gamma-nloglik")
.describe("Negative log-likelihood for gamma regression.")
.set_body([](const char* param) { return new EvalGammaNLogLik(); });

XGBOOST_REGISTER_METRIC(TweedieNLogLik, "tweedie-nloglik")
.describe("tweedie-nloglik@rho for tweedie regression.")
.set_body([](const char* param) {
  return new EvalTweedieNLogLik(param);
});
  XGBOOST_REGISTER_METRIC(CIndex, "CIndex")
                .describe("Concordance index.")
                .set_body([](const char* param) { return new EvalCIndex(); });

        XGBOOST_REGISTER_METRIC(PartialLikelihood, "partial_likelihood")
                .describe("Cox Proprotional Hazard Partial Likelihood.")
                .set_body([](const char* param) { return new EvalPartialLikelihood(); });



}  // namespace metric
}  // namespace xgboost

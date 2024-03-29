#ifndef TUNNINGPOINTPAIRSFACTOR_H
#define TUNNINGPOINTPAIRSFACTOR_H

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam
{

typedef Eigen::Matrix<double, 3, 6> M36;
template<class VALUE>
class TunningPointPairsFactor : public NoiseModelFactor2<VALUE, VALUE>
{
public:

  typedef VALUE T;
private:

  typedef TunningPointPairsFactor<VALUE> This;
  typedef NoiseModelFactor2<VALUE, VALUE> Base;

public:
    /**
     * @brief Construct a TunningPointPairsFactor
     */
    TunningPointPairsFactor(Key key1, Key key2, const Point3& p1, const Point3& p2,
                            const SharedNoiseModel& model = nullptr)
      : Base(model, key1, key2), p1_(p1), measured_(p2)
    {
      skewsym_p1_ = skewSymmetric(-p1_);
    }

    TunningPointPairsFactor()
    {
    }

    ~TunningPointPairsFactor()
    {
    }

    /// print with optional string
    void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
      std::cout << s << "TunningPointPairsFactor("
          << keyFormatter(this->key1()) << ","
          << keyFormatter(this->key2()) << ")\n";
      traits<Point3>::Print(measured_, "  measured: ");
      this->noiseModel_->print("  noise model: ");
    }

    Vector evaluateError(const T& T1, const T& T2, boost::optional<Matrix&> H1 =
        boost::none,boost::optional<Matrix&> H2 = boost::none) const override
    {
      const Matrix3 R2_tps = T2.rotation().transpose().matrix();
      const Matrix3 tmp_mat = R2_tps*T1.rotation().matrix();
      const Point3 p1_tran = tmp_mat*p1_+R2_tps*(T1.translation()-T2.translation());
      if (H1)
      {
        Matrix36 H1_tmp_;
//        float determ = fabs(tmp_mat.maxCoeff());
        const Matrix3 tmp_mat2 = tmp_mat*skewsym_p1_;
        H1_tmp_ << tmp_mat2, tmp_mat;
        *H1 = H1_tmp_;
      }
      if (H2)
      {
        Matrix36 H2_tmp_;
        const Matrix3 p1tran_skew = skewSymmetric(p1_tran);
        H2_tmp_ <<  p1tran_skew, -I_3x3;
        *H2 = H2_tmp_;
      }
      return p1_tran - measured_;
    }

    const Point3& measured() const {
      return measured_;
    }


private:
//    Vector3 p1_;
    Point3 p1_;
    Point3 measured_;
    Matrix3 skewsym_p1_;

    /** Serialization function */
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & boost::serialization::make_nvp("NoiseModelFactor2",
          boost::serialization::base_object<Base>(*this));
      ar & BOOST_SERIALIZATION_NVP(p1_) & BOOST_SERIALIZATION_NVP(measured_);
    }

};

}  // namespace gtsam
#endif // TUNNINGPOINTPAIRSFACTOR_H

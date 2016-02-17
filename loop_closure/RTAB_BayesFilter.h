#ifndef RTABMAP_BAYESFILTER_H_
#define RTABMAP_BAYESFILTER_H_

class RTABMAP_BayesFilter
{
public:
	BayesFilter(const ParametersMap & parameters = ParametersMap());
	virtual ~BayesFilter();
	virtual void parseParameters(const ParametersMap & parameters);
	const std::map<int, float> & computePosterior(const Memory * memory, const std::map<int, float> & likelihood);
	void reset();

	//setters
	void setPredictionLC(const std::string & prediction);

	//getters
	const std::map<int, float> & getPosterior() const {return _posterior;}
	float getVirtualPlacePrior() const {return _virtualPlacePrior;}
	const std::vector<double> & getPredictionLC() const; // {Vp, Lc, l1, l2, l3, l4...}
	std::string getPredictionLCStr() const; // for convenience {Vp, Lc, l1, l2, l3, l4...}

	cv::Mat generatePrediction(const Memory * memory, const std::vector<int> & ids) const;

	bool defaultBayesFullPredictionUpdate=true;
	std::vector<double> defaultBayesPredictionLC={0.1, 0.24, 0.18, 0.18, 0.1, 0.1, 0.04, 0.04, 0.01, 0.01};

private:
	cv::Mat updatePrediction(const cv::Mat & oldPrediction,
			const Memory * memory,
			const std::vector<int> & oldIds,
			const std::vector<int> & newIds) const;
	void updatePosterior(const Memory * memory, const std::vector<int> & likelihoodIds);
	float addNeighborProb(cv::Mat & prediction,
			unsigned int col,
			const std::map<int, int> & neighbors,
			const std::map<int, int> & idToIndexMap) const;
	void normalize(cv::Mat & prediction, unsigned int index, float addedProbabilitiesSum, bool virtualPlaceUsed) const;

private:
	std::map<int, float> _posterior;
	cv::Mat _prediction;
	float _virtualPlacePrior;
	std::vector<double> _predictionLC; // {Vp, Lc, l1, l2, l3, l4...}
	bool _fullPredictionUpdate;
	float _totalPredictionLCValues;

};

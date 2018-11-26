#ifndef H_LOSS_FUNC
#define H_LOSS_FUNC

class LossFunc
{
public:
	float (*func)(float, float) = nullptr;
	float (*derivative)(float, float) = nullptr;
};

#endif
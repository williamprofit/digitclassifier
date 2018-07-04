#ifndef H_ACTIVATION_FUNC
#define H_ACTIVATION_FUNC

class ActivationFunc
{
public:
    float (*func)(float) = nullptr;
    float (*derivative)(float) = nullptr;
};

#endif

std::shared_ptr<impl::FFTBase<float>> GetDispatchImpl(int n, float);
std::shared_ptr<impl::FFTBase<double>> GetDispatchImpl(int n, double);

std::shared_ptr<impl::FFTVertBase<float>> GetVertDispatchImpl(int n, float);
std::shared_ptr<impl::FFTVertBase<double>> GetVertDispatchImpl(int n, double);

std::shared_ptr<impl::FFTDITBase<float>> GetDITDispatchImpl(int n, float);
std::shared_ptr<impl::FFTDITBase<double>> GetDITDispatchImpl(int n, double);

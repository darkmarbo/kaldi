
记录:
========================================  ========================================
test/gdb_test.sh 
    测试脚本 其中:
    v0-kaldi.bak 是原始kaldi
    v5 修改为多线程的
    online2-wav-nnet2-latgen-faster.cc.v1.test-endpoint   测试 endpoint的 
    --do-endpointing=true 等于true的时候  会对输入语音进行截断 
        比如118.wav 开始静音特别长，直接截断了.




========================================  server client  ========================================
1.拷贝 onlinebin 下的 server和client程序到 online2bin下  
2.按照onlinebin下的Makefile 修改Makefile 原始保存为bak
3.WaveData 根据wav_path 直接读取出data数组 
4.online/online-tcp-source.cc

5. 多线程 socket 同时连接 解码 
6. socket接收data的形式 

========================================   online2bin  ========================================
ok:	struct 结构体中包含string ， 构造结构体的时候  使用new 而非malloc  不然string类型赋值的时候出core  最后还是把string换成 char ttt[MAX] 
ok:	访问全局变量g_wav_reader的时候未加锁，导致core  
ok:	输入的wavlist  获取utt的序号  然后利用 g_wav_reader 获取wave_data 
ok:	解决load_task 时  把所有语音的数据结构都load到内存了  占用较大 
ok:	时间花费 计算 定义一个全局统计 然后每次访问进行mutex  

================================  代码  =================================================

ivector/voice-activity-detection.cc:  
    ivectorbin/compute-vad.cc   内部有使用方法
        SequentialBaseFloatMatrixReader   util/table-types.h 中  (key value) 
            typedef SequentialTableReader<KaldiObjectHolder<Matrix<BaseFloat> > >
                SequentialBaseFloatMatrixReader;
    

online2/online-nnet2-decoding.cc
    SingleUtteranceNnet2Decoder 类 
        其中 LatticeFasterOnlineDecoder decoder_;

onlinebin/online-audio-server-decode-faster.cc
	服务器形式  在线接受客户端的请求 
	http://kaldi-asr.org/doc/online_programs.html

online2-wav-nnet2-latgen-faster.cc
	目前使用的解码流程  
online2/online-nnet2-decoding.cc
	decoder 类 

online/online-faster-decoder.cc:
	decoder 类 

online/online-feat-input.h
	OnlineFeInput 类
	从 OnlineAudioSource 中读取wav数据 提取MFCC/PLP特征 
	int32 samples_req = frame_size_ + (nvec - 1) * frame_shift_;
	Vector<BaseFloat> read_samples(samples_req);
	bool ans = source_->Read(&read_samples);
		使用 	typedef OnlineFeInput<Mfcc> FeInput;
		 	FeInput fe_input(au_src, &mfcc, 25ms帧长度|窗长度,帧移);

online/online-tcp-source.cc  server端 socket接收的类 
	OnlineTcpVectorSource 类
		使用 au_src = new OnlineTcpVectorSource(client_socket)
		bool Read(Vector<BaseFloat> *data)

util/kaldi-table.h  RandomAccessTableReader 类的声明  
util/kaldi-table-inl.h
		typedef typename Holder::T T; // 就是WaveData  
			主要成员 WaveHolder 类对象
		RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
			oneline2bin中 使用时的例子 
		explicit RandomAccessTableReader(const std::string &rspecifier);
			构造函数 接收file_name 内部调用了Open 
		RandomAccessTableReaderImplBase<Holder> *impl_;
			主要成员 另一个接口  内部的其他函数都是针对它  
		bool Open(const std::string &rspecifier);
			分类型new出不同的class  赋值给 impl_  	
		bool HasKey(const std::string &key);
			判断是否有 key 
		const T &Value(const std::string &key);
			得到 WaveHolder::T 也就是 WaveData  
		std::vector<std::pair<std::string, std::string> > script_;
			主要成员 
		return holder_.Value();
			其中 Holder holder_;   Holder是魔板传进来的 
	

	WaveData wave_data = g_wav_reader.Value(utt);
	// 读取wav的channel--0
	SubVector<BaseFloat> data(wave_data.Data(), 0);
	BaseFloat samp_freq = wave_data.SampFreq();

feat/wave-reader.h
	WaveHolder 类 
		typedef WaveData T;  
		T t_;
			主要成员
		bool Read(std::istream &is) {
			读取 wave 文件流 到WaveData t_;
		const T &Value() { return t_; }
			返回 WaveData t_; 

	WaveData 类
		WaveData(BaseFloat samp_freq, const MatrixBase<BaseFloat> &data): data_(data), samp_freq_(samp_freq) {}
			使用采样率和 data 初始化 
		void Read(std::istream &is, ReadDataType read_data = kReadData);
			读取wave文件流  
		const Matrix<BaseFloat> &Data() const { return data_; }
			返回 data 数组 
		BaseFloat SampFreq() const { return samp_freq_; }
			返回 samp 采样率 
		BaseFloat Duration() const { return data_.NumCols() / samp_freq_; }
			返回 wave 的时长 
		Matrix<BaseFloat> data_;
			主要成员 data 数组 

		

online2/online-nnet2-feature-pipeline.h

	class  OnlineNnet2FeaturePipeline 和 OnlineNnet2FeaturePipelineInfo
  
	/// Accept more data to process.  It won't actually process it until you call
  	/// GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you
  	/// call this function it will just copy it).  sampling_rate is necessary just
  	/// to assert it equals what's in the config.
  	void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

================================  测试  =================================================
编译 onlinebin 下的程序  需要执行:  make ext 
过程中 缺少Portaudio 需要在 tools 执行相应脚本 安装(下载失败 手动下载)
	linux下wget下载的 不能tar解压   windows上下载的可以  






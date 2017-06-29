// online2bin/online2-wav-nnet2-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"

using namespace kaldi;
using namespace fst;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

#define NUM_THREADS 50
#define NUM_MAX_PATH 1025

int32 g_num_done = 0; // 处理完成语音总数 
int32 g_num_err = 0;  // 问题语音总数 
int64 g_num_frames = 0; // 总帧数 
double g_tot_like = 0.0; // 概率和 
int	g_speech_time;//ms

//互斥锁  定义全局变量，让所有线程同时写，这样就需要锁机制
pthread_mutex_t g_asr_cmd_lock;


struct decoder_data_t{
	const OnlineNnet2FeaturePipelineInfo *feature_info;
	const OnlineNnet2DecodingConfig *nnet2_decoding_config;
	const TransitionModel *trans_model;
	const nnet2::AmNnet *nnet;
	const fst::Fst<fst::StdArc> *decode_fst;;
	int id;
	BaseFloat chunk_length_secs;
	const fst::SymbolTable *word_syms;
	std::string clat_wspecifier;

};

struct task_t{
	char utt[NUM_MAX_PATH];  // 1	
	int nID; // wav的id编号 

};

struct task_data_t{
	int nTaskNum;      // 任务总数
	int nAllocNum;      // 开辟的数组大小 
	int nDealed;       // 目前处理完任务数 
	struct task_t *pAllTask; // 所有任务
	std::string* pAllTaskResult;  // 任务对应的识别结果 

};

struct decoder_data_t g_dec_data;
struct task_data_t g_task_data;

/*
 * 读取file-list 到vector中 
 * */
int read_wav_list(const std::string &file_list, std::vector<std::string> &vec_list)
{
	int ret = 0;
	std::string line;
	ifstream ifs(file_list.c_str());

	if(!ifs)
	{
		KALDI_ERR<<"open file "<<file_list<<" failed!\n";
		return -1;

	}

	while(ifs>>line)
	{

		KALDI_LOG<<"readline:"<<line;
	}

	ifs.close();
	ifs.clear();

	return ret;
}

/*
 * wav_list: spk_1 wav1  
 * taskdata: 任务列表 
 * */
int load_task(const std::string &file_list, task_data_t &taskdata)
{
	int ret = 0;
	std::string line;
	taskdata.nTaskNum =0;

	// 开辟初始空间 
	taskdata.nAllocNum = 50000;
	taskdata.pAllTask = (task_t*)malloc(sizeof(task_t)*taskdata.nAllocNum);
	taskdata.pAllTask = new task_t[taskdata.nAllocNum];
	if(taskdata.pAllTask == NULL)
	{
		KALDI_ERR << "ERROR: malloc taskdata.pAllTask failed!\n";
		return -1;
	}

	ifstream ifs(file_list.c_str());
	if(!ifs)
	{
		KALDI_ERR<<"open file "<<file_list<<" failed!\n";
		return -1;

	}

	while(ifs>>line)
	{

		KALDI_LOG<<"readline:"<<line;
		// 是否需要开辟 task_data_t 内的数组空间 
		if(taskdata.nTaskNum > taskdata.nAllocNum)
		{
			taskdata.nAllocNum *= 2;
			taskdata.pAllTask = (task_t*)realloc(taskdata.pAllTask, sizeof(task_t)*taskdata.nAllocNum);
			if(taskdata.pAllTask == NULL)
			{
				KALDI_ERR << "ERROR: realloc taskdata.pAllTask failed!\n";
				return -1;
			}
		}

	
		snprintf(taskdata.pAllTask[taskdata.nTaskNum].utt, NUM_MAX_PATH, "%s", line.c_str());
		taskdata.pAllTask[taskdata.nTaskNum].nID = taskdata.nTaskNum;
		taskdata.nTaskNum ++;
	}

	taskdata.pAllTaskResult = (std::string*)malloc(sizeof(std::string)*taskdata.nTaskNum);
	if(taskdata.pAllTaskResult == NULL)
	{
		KALDI_ERR << "ERROR: malloc taskdata.pAllTaskResult failed!\n";
		return -1;
	}
	memset(taskdata.pAllTaskResult, 0, sizeof(std::string)*taskdata.nTaskNum);

	ifs.close();
	ifs.clear();

	return ret;

}

task_t* fetch_task()
{
	task_t* task;
	pthread_mutex_lock(&g_asr_cmd_lock);
	if(g_task_data.nDealed >= g_task_data.nTaskNum)
	{
		pthread_mutex_unlock(&g_asr_cmd_lock);
		return NULL;	
	}
	task = g_task_data.pAllTask + g_task_data.nDealed++;
	pthread_mutex_unlock(&g_asr_cmd_lock);
	return task;
}


/*
 * 统计每个语音的信息 加和到全局变量中  
 * */
void GetDiagnosticsAndPrintOutput( const std::string &utt,const fst::SymbolTable *word_syms,
		const CompactLattice &clat, int64 *tot_num_frames, double *tot_like) 
{
	if (clat.NumStates() == 0) 
	{
		KALDI_WARN << "Empty lattice.";
		return;
	}
	// 从压缩格式的lat中获取one_best 然后转换成Lattice 
	CompactLattice best_path_clat;
	CompactLatticeShortestPath(clat, &best_path_clat);

	Lattice best_path_lat;
	ConvertLattice(best_path_clat, &best_path_lat);

	double likelihood;
	LatticeWeight weight;
	int32 num_frames;
	std::vector<int32> alignment;
	std::vector<int32> words; // 每个word 对应的ID  

	GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
	num_frames = alignment.size();
	likelihood = -(weight.Value1() + weight.Value2());
	*tot_num_frames += num_frames;
	*tot_like += likelihood;

	KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
		<< (likelihood / num_frames) << " over " << num_frames
		<< " frames.";

	if (word_syms != NULL) 
	{
		std::cerr << utt << '\t';
		for (size_t i = 0; i < words.size(); i++) 
		{
			std::string s = word_syms->Find(words[i]);
			if (s == "")
			{
				KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";

			}
			//std::cerr << s << ' ';
			std::cerr << s;
		}
		std::cerr << std::endl;
	}
}


/*
 * 识别一个wav  
 *
 * */
int recog_one_audio(const decoder_data_t *p_dec_data, const task_t *task)
{

	int 	ret = 0;
	struct	timeval start_time;
	struct	timeval end_time;
	int	cpu_time;//ms
	int	speech_time;//ms
	float	rt_factor;
	int32	chunk_length;
	int32	samp_offset = 0;

	std::string utt(task->utt);

	gettimeofday(&start_time, NULL);

	// 创建decoder  
	OnlineNnet2FeaturePipeline feature_pipeline(*(p_dec_data->feature_info));
	SingleUtteranceNnet2Decoder decoder(*p_dec_data->nnet2_decoding_config, *p_dec_data->trans_model, 
			*p_dec_data->nnet, *p_dec_data->decode_fst, &feature_pipeline);


	// 不加锁  出core 
	pthread_mutex_lock(&g_asr_cmd_lock);

	WaveData wave_data;
	std::filebuf fb;
	fb.open(utt.c_str(), std::ios::in);
	std::istream is(&fb);
	wave_data.Read(is);
	pthread_mutex_unlock(&g_asr_cmd_lock);

	// 读取wav的channel--0
	SubVector<BaseFloat> data(wave_data.Data(), 0);
	// 读取wav的采样率 
	BaseFloat samp_freq = wave_data.SampFreq();

	if (p_dec_data->chunk_length_secs > 0) 
	{
		chunk_length = int32(samp_freq * p_dec_data->chunk_length_secs);
		if (chunk_length == 0) chunk_length = 1;
	} 
	else 
	{
		chunk_length = std::numeric_limits<int32>::max();
	}

	speech_time = (1000*data.Dim())/samp_freq;
	// 语音的 每一段 
	while (samp_offset < data.Dim()) 
	{
		int32 samp_remaining = data.Dim() - samp_offset;
		int32 num_samp = chunk_length < samp_remaining ? chunk_length
			: samp_remaining;

		SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
		feature_pipeline.AcceptWaveform(samp_freq, wave_part);

		samp_offset += num_samp;

		// 核心解码 
		decoder.AdvanceDecoding();

	}

	// 整个语音解码完成  
	decoder.FinalizeDecoding();

	// 获取lattice 
	CompactLattice clat;
	bool end_of_utterance = true;
	decoder.GetLattice(end_of_utterance, &clat);

	// 加锁 计算全局统计信息 
	pthread_mutex_lock(&g_asr_cmd_lock);
	// 计算全局统计信息 并且输出log  
	g_speech_time += speech_time;
	GetDiagnosticsAndPrintOutput(utt, p_dec_data->word_syms, clat, &g_num_frames, &g_tot_like);
	g_num_done++;
	pthread_mutex_unlock(&g_asr_cmd_lock);

	// 计算实时率
	gettimeofday(&end_time, NULL);
	cpu_time = (end_time.tv_sec - start_time.tv_sec)*1000 + (end_time.tv_usec - start_time.tv_usec)/1000;
	rt_factor = (float)cpu_time/speech_time;
	KALDI_LOG<<"cpu_time="<<cpu_time<<"\tspeech_time="<<speech_time<<"\trt_factor="<<rt_factor<<"\n";


	//// we want to output the lattice with un-scaled acoustics.
	//BaseFloat inv_acoustic_scale =
	//	1.0 / p_dec_data->nnet2_decoding_config->decodable_opts.acoustic_scale;
	//ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);


	return ret;


}

// 多线程解码函数 
void* fun_decode(void *args)
{

	int ret = 0;
	task_t *task;
	// 解码资源 
	decoder_data_t *p_dec_data = (decoder_data_t *)args;

	//  不断的取task  进行识别   
	while((task = fetch_task()) != NULL)
	{	
		ret = recog_one_audio(p_dec_data, task);
		if(ret < 0)
		{
			KALDI_ERR<<"recog_one_audio failed!\n";
			return NULL;
		}

	}

	pthread_exit( 0 );

}



int main(int argc, char *argv[]) 
{
	try {

		const char *usage =
			"Reads in wav file(s) and simulates online decoding with neural nets\n"
			"(nnet2 setup), with optional iVector-based speaker adaptation and\n"
			"optional endpointing.  Note: some configuration values and inputs are\n"
			"set via config files whose filenames are passed as options\n"
			"\n"
			"Usage: online2-wav-nnet2-latgen-faster [options] <nnet2-in> <fst-in> "
			"<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
			"The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
			"you want to decode utterance by utterance.\n"
			"See egs/rm/s5/local/run_online_decoding_nnet2.sh for example\n"
			"See also online2-wav-nnet2-latgen-threaded\n";


		int ret = 0;

		ParseOptions po(usage);

		// 基础变量定义 
		std::string word_syms_rxfilename; // words.txt
		OnlineEndpointConfig endpoint_config;
		// feature_config includes configuration for the iVector adaptation,
		// as well as the basic features.
		OnlineNnet2FeaturePipelineConfig feature_config;  
		OnlineNnet2DecodingConfig nnet2_decoding_config;
		BaseFloat chunk_length_secs = 0.05;
		bool do_endpointing = false; // true
		bool online = true; // true

		po.Register("chunk-length", &chunk_length_secs,
				"Length of chunk size in seconds, that we process.  Set to <= 0 "
				"to use all input in one chunk.");
		po.Register("word-symbol-table", &word_syms_rxfilename,
				"Symbol table for words [for debug output]");
		po.Register("do-endpointing", &do_endpointing,
				"If true, apply endpoint detection");
		po.Register("online", &online,
				"You can set this to false to disable online iVector estimation "
				"and have all the data for each utterance used, even at "
				"utterance start.  This is useful where you just want the best "
				"results and don't care about online operation.  Setting this to "
				"false has the same effect as setting "
				"--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
				"in the file given to --ivector-extraction-config, and "
				"--chunk-length=-1.");
		po.Register("num-threads-startup", &g_num_threads,
				"Number of threads used when initializing iVector extractor.");

		feature_config.Register(&po);
		nnet2_decoding_config.Register(&po);
		endpoint_config.Register(&po);

		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
			po.PrintUsage();
			return 1;
		}

		// 参数 传递 
		std::string nnet2_rxfilename = po.GetArg(1),
			fst_rxfilename = po.GetArg(2),
			spk2utt_rspecifier = po.GetArg(3),
			wav_rspecifier = po.GetArg(4),
			clat_wspecifier = po.GetArg(5);
		//std::string scp_list(wav_rspecifier,4,wav_rspecifier.size()-4); // dir_wav.testset.list



		OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
		if (!online) 
		{
			feature_info.ivector_extractor_info.use_most_recent_ivector = true;
			feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
			chunk_length_secs = -1.0;
		}

		/////////////////////////////////////////   load 模型   ///////////////////////////////////////////////////
		// 读取模型 
		KALDI_LOG << "start read model...\n";
		// 声学模型 mdl 
		TransitionModel trans_model;
		nnet2::AmNnet nnet;
		{
			bool binary;
			Input ki(nnet2_rxfilename, &binary);
			trans_model.Read(ki.Stream(), binary);
			nnet.Read(ki.Stream(), binary);
		}

		// fst 
		fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);

		// graph_pp_tgsmall/words.txt 
		fst::SymbolTable *word_syms = NULL;
		if (word_syms_rxfilename != "")
			if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
				KALDI_ERR << "Could not read symbol table from file "
					<< word_syms_rxfilename;


		// 解码资源 全局变量赋值  
		g_dec_data.feature_info = &feature_info;
		g_dec_data.nnet2_decoding_config = &nnet2_decoding_config;
		g_dec_data.trans_model = &trans_model;
		g_dec_data.nnet = &nnet;
		g_dec_data.decode_fst = decode_fst;
		g_dec_data.chunk_length_secs = chunk_length_secs;
		g_dec_data.word_syms = word_syms;
		g_dec_data.clat_wspecifier = clat_wspecifier;

		KALDI_LOG << "end read model...\n";

		// 读取测试集list  到 taskdata  中 
		//bzero(&g_task_data, sizeof(task_data_t));
		ret = load_task(wav_rspecifier, g_task_data);
		if(ret < 0)
		{
			KALDI_ERR<<"load_task failed!\n";
			return -1;
		}

		// 全部语音处理 时间信息统计 
		struct	timeval main_start_time;
		struct	timeval main_end_time;
		int	main_cpu_time;//ms
		float	main_rt_factor;
		gettimeofday(&main_start_time, NULL);

		//////////////////////////////////////////////////////   多线程  /////////////////////////////////////////////////////////
		pthread_t tids[NUM_THREADS];

		//线程属性结构体，创建线程时加入的参数
		pthread_attr_t attr;
		//初始化
		pthread_attr_init( &attr );
		//是设置你想要指定线程属性参数，这个参数表明这个线程是可以join连接的，join功能表示主程序可以等线程结束后再去做某事，>
		//实现了主程序和线程同步功能
		pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
		//对锁进行初始化    
		pthread_mutex_init( &g_asr_cmd_lock, NULL );


		KALDI_LOG << "test:start 创建线程 解码 !\n" ;
		for( int i = 0; i < NUM_THREADS; ++i )
		{
			g_dec_data.id = i;
			// 解码线程创建 
			int ret = pthread_create( &tids[i], &attr, fun_decode, ( void* )&( g_dec_data ) );
			if( ret != 0 )
			{
				cout << "pthread_create error:error_code=" << ret << endl;
			}
		}

		//释放内存 
		pthread_attr_destroy( &attr );
		void *status;
		for( int i = 0; i < NUM_THREADS; ++i )
		{
			//主程序join每个线程后取得每个线程的退出信息status
			int ret = pthread_join( tids[i], &status );
			if( ret != 0 )
			{
				cout << "pthread_join error:error_code=" << ret << endl;
			}
		}
		KALDI_LOG << "decoding completed!\n" ;

		pthread_mutex_destroy( &g_asr_cmd_lock ); //注销锁

		KALDI_LOG << "Decoded " << g_num_done << " utterances, "
			<< g_num_err << " with errors.";
		KALDI_LOG << "Overall likelihood per frame was " << (g_tot_like / g_num_frames)
			<< " per frame over " << g_num_frames << " frames.";

		// 计算实时率
		gettimeofday(&main_end_time, NULL);
		main_cpu_time = (main_end_time.tv_sec - main_start_time.tv_sec)*1000 + (main_end_time.tv_usec - main_start_time.tv_usec)/1000;
		main_rt_factor = (float)main_cpu_time/g_speech_time;
		KALDI_LOG<<"main_cpu_time="<<main_cpu_time<<"\tg_speech_time="<<g_speech_time<<"\tg_rt_factor="<<main_rt_factor<<"\n";


		delete decode_fst;
		delete word_syms; // will delete if non-NULL.
		return (g_num_done != 0 ? 0 : 1);



	} // end main  
	catch(const std::exception& e) 
	{
		std::cerr << e.what();
		return -1;
	}


} // main()








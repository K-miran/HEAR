#ifndef THREAD_H_
#define THREAD_H_

#include <thread>
#include <condition_variable>

#include <atomic>
#include <mutex>

#include <vector>
#include <future>
#include <assert.h>

#define DEFAULT_NUM_THREADS 4

namespace Thread {

class ThreadPool {
private:

   // Simple Signal
   template<class T>
   class SimpleSignal {
   private:
     T val; 
     std::mutex m;
     std::condition_variable cv;
   
     SimpleSignal(const SimpleSignal&) = delete;
     void operator=(const SimpleSignal&) = delete;
   
   public:
     SimpleSignal() : val(0) { }
   
     T wait() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return val; } );
        T old_val = val;
        val = 0;
        return old_val;
     }
   
     void send(T new_val) {
        std::lock_guard<std::mutex> lock(m);
        val = new_val;
        cv.notify_one();
     }
   };
   
   // Composite Signal
   template<class T, class T1>
   class CompositeSignal {
   private:
     T val; 
     T1 val1;
     std::mutex m;
     std::condition_variable cv;
   
     CompositeSignal(const CompositeSignal&);
     void operator=(const CompositeSignal&);
   
   public:
     CompositeSignal() : val(0) { }
   
     T wait(T1& _val1) {
       std::unique_lock<std::mutex> lock(m);
       cv.wait(lock, [&]() { return val; } );
       T _val = val;
       _val1 = val1;
       val = 0;
       return _val;
     }
   
     void send(T _val, T1 _val1) {
       std::lock_guard<std::mutex> lock(m);
       val = _val;
       val1 = _val1;
       cv.notify_one();
     }
   };

   // Task
   class Task {
   private:
      ThreadPool * pool_;
   public:
      Task(ThreadPool * pool)
         : pool_(pool) {
         // nothing to do
      }
      virtual ~Task() { }
      ThreadPool * getThreadPool() { return pool_; }
      virtual void run(long index) = 0;
   };

   class TerminateTask : public Task {
   public:
      TerminateTask() : Task(0) { }
      void run(long index) { }
   };

   
   template<class F>
   class FunctionTask : public Task {

   public:
      F const& f;

      long nsubtasks;  // number of subproblems
      long subtasksz;  // interval size of large subproblems
      long nsmalls;    // number of small subproblems

      FunctionTask(ThreadPool * _pool, F const& _f, size_t _sz, size_t _nt)
         : Task(_pool), f(_f) {
         if(_sz < _nt) {
            nsubtasks = _sz;
            subtasksz = 1;
            nsmalls = 0;
         } else {
            nsubtasks = _nt;
            long q = _sz/_nt;
            long r = _sz - _nt*q;
            if(r == 0) {
               subtasksz = q;
               nsmalls = 0;
            } else {
               subtasksz = q+1;
               nsmalls = _nt - r;
            }
         }
      }
      ~FunctionTask() { }

      void run(long i) {
         long first, last;
         if(i < nsmalls) {  // small interval
            first = i*(subtasksz-1);
            last  = first + (subtasksz-1);
         } else {
            first = nsmalls*(subtasksz-1) + (i-nsmalls)*subtasksz;
            last = first + subtasksz;
         }
         f(first, last);
      }
   };

   struct WorkerThread {
      CompositeSignal< Task *, long > localSignal;
      TerminateTask term;
      std::thread t;
   
      WorkerThread() : t(worker, &localSignal) { 
         // nothing to do
      }
   
      ~WorkerThread() {
         localSignal.send(&term, -1);
         t.join();
      }
   };

   std::vector<WorkerThread *> thread_;

   SimpleSignal<bool> globalSignal_;

   std::atomic<long> active_threads_;

   ThreadPool(const ThreadPool&) = delete;
   void operator=(const ThreadPool&) = delete;

   void launch(Task *task, long index) {
      thread_[index-1]->localSignal.send(task, index);
   }

   void begin(long cnt) {
      active_threads_ = cnt;
   }

   void end() {
      globalSignal_.wait();
   }

   static void runOneTask(Task *task, long index) {
      ThreadPool * pool = task->getThreadPool();
      task->run(index);
      if(--(pool->active_threads_) == 0) {
         pool->globalSignal_.send(true);
      }
   }

   static void worker(CompositeSignal< Task *, long > * localSignal) {
      for (;;) {
         long index = -1;
         Task *task = localSignal->wait(index);
         if(index == -1) {
            return;
         }
         runOneTask(task, index);
      }
   }

public:

   explicit ThreadPool(size_t n)
      : active_threads_(0) {
      for(size_t i = 1; i < n; i++) { // create n-1 threads
         WorkerThread * t = new WorkerThread();
         thread_.push_back(t);
      }
   }
         
   ~ThreadPool() {
      if(active()) {
         assert(0&&"Destructed while active");
      }
      while(!thread_.empty()) {
         WorkerThread * t = thread_.back();
         thread_.pop_back();
         delete t;
      }
   }

   long numThreads() const {
      return thread_.size()+1;  // including the current thread
   }
   bool active() const {
      return (active_threads_>0);
   }

   template<class Fct>
   void exec_range(long sz, const Fct& fct) {
      if(active()) {
         assert(0&&"ThreadPool: illegal operation while active");
      }
      if(sz <= 0) {
         return;
      }
      long nt = numThreads();

      FunctionTask<Fct> task(this, fct, sz, nt);

      begin(task.nsubtasks);
      for(long i = 1; i < task.nsubtasks; i++) {
         launch(&task, i);      // prepare other threads
      }
      runOneTask(&task, 0);     // run the first subproblem on this thread
      end();
   }
};

template<class Fct>
static void relaxed_exec_range(ThreadPool *pool, long sz, const Fct& fct) {
   if(sz <= 0) {
      return;
   }
   if(!pool || pool->active() || sz == 1) {
      fct(0, sz);
   } else {
      pool->exec_range(sz, fct);
   }
}

extern
ThreadPool *threadPool_ptr__;

inline ThreadPool *getThreadPool() {
   return threadPool_ptr__;
}

void initThreadPool(size_t n);

inline long availableThreads() {
   ThreadPool *pool = getThreadPool();
   if(!pool || pool->active()) {
      return 1;
   } else {
      return pool->numThreads();
   }
}

}

#define MT_EXEC_RANGE_(n, first, last){  \
   Thread::relaxed_exec_range(Thread::getThreadPool(), (n),  \
      [&](long first, long last) {  \

#define MT_EXEC_RANGE_END_\
    } );\
}\

#define MT_EXEC_RANGE(n, first, last)  \
   MT_EXEC_RANGE_((n),(first),(last))

#define MT_EXEC_RANGE_END  \
   MT_EXEC_RANGE_END_

#endif

/**
 * Queues - Queue Pair
 *
 * (c) 2018 Claude Barthels, ETH Zurich
 * Contact: claudeb@inf.ethz.ch
 *
 */

#ifndef QUEUES_QUEUEPAIR_H_
#define QUEUES_QUEUEPAIR_H_

#include <infiniband/verbs.h>

#include <infinity/core/Context.h>
#include <infinity/memory/Atomic.h>
#include <infinity/memory/Buffer.h>
#include <vector>
#include <infinity/memory/RegionToken.h>
#include <infinity/requests/RequestToken.h>

namespace infinity {
namespace queues {
	class QueuePairFactory;
}
}

namespace infinity {
namespace queues {
struct SendRequestBuffer {
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> requests;
  SendRequestBuffer(){}
  SendRequestBuffer(int num) {
    sges.resize(num);
    requests.resize(num);
  }
  void resize(int num){
	sges.resize(num);
    requests.resize(num);
  }
  void reset() {
    memset(sges.data(), 0, sizeof(ibv_sge));
    memset(requests.data(), 0, sizeof(ibv_send_wr));
  }
};
} // namespace queues
}

namespace infinity {
namespace queues {

struct IbvWcBuffer {
  ibv_wc* wc;
  int size_;
  IbvWcBuffer(){}
  IbvWcBuffer(int size) {
	wc = (ibv_wc*) malloc(sizeof(ibv_wc) * size);
	size_ = size;
  }
  void resize(int size){
	wc = (ibv_wc*) malloc(sizeof(ibv_wc) * size);
	size_ = size;
  }

  ibv_wc* ptr(){
	  return wc;
  }
  int size(){
	  return size_;
  }

};
} // namespace queues
}


namespace infinity {
namespace queues {

class OperationFlags {

public:
  bool fenced;
  bool signaled;
  bool inlined;

  OperationFlags() : fenced(false), signaled(false), inlined(false) { };

  /**
   * Turn the bools into a bit field.
   */
  int ibvFlags();
};

class QueuePair {

	friend class infinity::queues::QueuePairFactory;

public:

	/**
	 * Constructor
	 */
	QueuePair(infinity::core::Context *context);

	/**
	 * Destructor
	 */
	~QueuePair();

protected:

	/**
	 * Activation methods
	 */

	void activate(uint16_t remoteDeviceId, uint32_t remoteQueuePairNumber, uint32_t remoteSequenceNumber);
	void setRemoteUserData(void *userData, uint32_t userDataSize);

public:

	/**
	 * User data received during connection setup
	 */

	bool hasUserData();
	uint32_t getUserDataSize();
	void * getUserData();

public:

	/**
	 * Queue pair information
	 */

	uint16_t getLocalDeviceId();
	uint32_t getQueuePairNumber();
	uint32_t getSequenceNumber();

public:

	/**
	 * Buffer operations
	 */

	void send(infinity::memory::Buffer *buffer, infinity::requests::RequestToken *requestToken = NULL);
	void send(infinity::memory::Buffer *buffer, uint32_t sizeInBytes, infinity::requests::RequestToken *requestToken = NULL);
	void send(infinity::memory::Buffer *buffer, uint64_t localOffset, uint32_t sizeInBytes, OperationFlags flags,
      infinity::requests::RequestToken *requestToken = NULL);

	void write(infinity::memory::Buffer *buffer, infinity::memory::RegionToken *destination, infinity::requests::RequestToken *requestToken = NULL);
	void write(infinity::memory::Buffer *buffer, infinity::memory::RegionToken *destination, uint32_t sizeInBytes,
			infinity::requests::RequestToken *requestToken = NULL);
	void write(infinity::memory::Buffer *buffer, uint64_t localOffset, infinity::memory::RegionToken *destination, uint64_t remoteOffset, uint32_t sizeInBytes,
      OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

	void read(infinity::memory::Buffer *buffer, infinity::memory::RegionToken *source, infinity::requests::RequestToken *requestToken = NULL);
	void read(infinity::memory::Buffer *buffer, infinity::memory::RegionToken *source, uint32_t sizeInBytes, infinity::requests::RequestToken *requestToken =
	NULL);
	void read(infinity::memory::Buffer *buffer, uint64_t localOffset, infinity::memory::RegionToken *source, uint64_t remoteOffset, uint32_t sizeInBytes,
			OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

public:

	/**
	 * Complex buffer operations
	 */

	void multiWrite(infinity::memory::Buffer **buffers, uint32_t *sizesInBytes, uint64_t *localOffsets, uint32_t numberOfElements,
			infinity::memory::RegionToken *destination, uint64_t remoteOffset, OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);
	
	void multiRead(uint32_t batch_size, infinity::memory::Buffer *buffer, uint64_t* localOffset, infinity::memory::RegionToken *source, uint64_t* remoteOffset,
			uint32_t sizeInBytes, OperationFlags send_flags, infinity::requests::RequestToken *requestToken, infinity::queues::SendRequestBuffer &send_buffer);

	void sendWithImmediate(infinity::memory::Buffer *buffer, uint64_t localOffset, uint32_t sizeInBytes, uint32_t immediateValue,
			OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

	void writeWithImmediate(infinity::memory::Buffer *buffer, uint64_t localOffset, infinity::memory::RegionToken *destination, uint64_t remoteOffset,
			uint32_t sizeInBytes, uint32_t immediateValue, OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

	void multiWriteWithImmediate(infinity::memory::Buffer **buffers, uint32_t *sizesInBytes, uint64_t *localOffsets, uint32_t numberOfElements,
			infinity::memory::RegionToken *destination, uint64_t remoteOffset, uint32_t immediateValue, OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

public:

	/**
	 * Atomic value operations
	 */

	void compareAndSwap(infinity::memory::RegionToken *destination, uint64_t compare, uint64_t swap, infinity::requests::RequestToken *requestToken = NULL);
	void compareAndSwap(infinity::memory::RegionToken *destination, infinity::memory::Atomic *previousValue, uint64_t compare, uint64_t swap,
			OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);
	void fetchAndAdd(infinity::memory::RegionToken *destination, uint64_t add, infinity::requests::RequestToken *requestToken = NULL);
	void fetchAndAdd(infinity::memory::RegionToken *destination, infinity::memory::Atomic *previousValue, uint64_t add,
			OperationFlags flags, infinity::requests::RequestToken *requestToken = NULL);

protected:

	infinity::core::Context * const context;

	ibv_qp* ibvQueuePair;
	uint32_t sequenceNumber;

	void *userData;
	uint32_t userDataSize;

};

} /* namespace queues */
} /* namespace infinity */

#endif /* QUEUES_QUEUEPAIR_H_ */

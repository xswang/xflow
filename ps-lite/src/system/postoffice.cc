#include "system/postoffice.h"
#include "ps/app.h"

namespace ps {

DECLARE_string(interface);

Postoffice::~Postoffice() {
  if (recv_thread_) recv_thread_->join();
  if (send_thread_) {
    Message* stop = new Message(); stop->terminate = true; Queue(stop);
    send_thread_->join();
  }
}

void Postoffice::Run(int* argc, char*** argv) {
  google::InitGoogleLogging((*argv)[0]);
  google::ParseCommandLineFlags(argc, argv, true);
  manager_.Init(*argc, *argv);
  // start the I/O threads
  recv_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&Postoffice::Recv, this));
  send_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&Postoffice::Send, this));
  manager_.Run();
}

void Postoffice::Send() {
  Message* msg;
  while (true) {
    sending_queue_.wait_and_pop(msg);
    if (msg->terminate) break;
    size_t send_bytes = 0;
    manager_.van().Send(msg, &send_bytes);
    manager_.net_usage().IncrSend(msg->recver, send_bytes);
    if (msg->task.request()) {
      // a request "msg" is safe to be deleted only if the response is received
      manager_.AddRequest(msg);
    } else {
      delete msg;
    }
  }
}

void Postoffice::Recv() {
  while (true) {
    // receive a message
    Message* msg = new Message();
    size_t recv_bytes = 0;
    CHECK(manager_.van().Recv(msg, &recv_bytes));
    manager_.net_usage().IncrRecv(msg->sender, recv_bytes);
    // process
    if (!msg->task.request()) manager_.AddResponse(msg);
    if (msg->task.control()) {
      if (!manager_.Process(msg)) {
        delete msg; break;
      }
      delete msg;
    } else {
      int id = msg->task.customer_id();
      // let the executor to delete "msg"
      manager_.customer(id)->executor()->Accept(msg);
    }
  }
}

} // namespace ps

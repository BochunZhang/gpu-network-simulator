  template <typename T>
  T drain_read(const T *device)
  {
    if (device == nullptr) throw std::runtime_error("null GeDES drain state pointer");
    T host{};
    const auto status = cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return host;
  }

  template <typename T>
  void drain_queues(std::ofstream &out, const char *name,
                    const std::vector<GPUQueue<T> *> &queues)
  {
    if (queues.empty()) throw std::runtime_error("missing GeDES queue group");
    for (size_t i = 0; i < queues.size(); ++i)
    {
      const auto queue = drain_read(queues[i]);
      if (queue.queue_capacity <= 0 || queue.size < 0 || queue.size > queue.queue_capacity ||
          queue.head < 0 || queue.head >= queue.queue_capacity)
        throw std::runtime_error("invalid GeDES queue occupancy");
      out << name << ',' << i << ',' << queue.size << '\n';
    }
  }

  void Topology::WriteDrainAudit(const std::string &flow_output) const
  {
    static_assert(ENABLE_FATTREE_MODE && ENABLE_GPU_MEM_POOL && !ENABLE_CACHE,
                  "the audit covers the pinned FatTree GPU-pool mode without packet cache");
    std::ofstream queues(flow_output + ".queues.csv");
    std::ofstream connections(flow_output + ".connections.csv");
    queues.exceptions(std::ios::failbit | std::ios::badbit);
    connections.exceptions(std::ios::failbit | std::ios::badbit);
    queues << "group,index,entries\n";
    drain_queues(queues, "tcp_ingress", m_node_tcp_ingress_queues);
    drain_queues(queues, "tcp_egress", m_node_tcp_egress_queues);
    drain_queues(queues, "ipv4_ingress", m_node_ipv4_ingress_queues);
    drain_queues(queues, "ipv4_egress", m_node_ipv4_egress_queues);
    drain_queues(queues, "ipv4_local", m_node_ipv4_local_delivery_queues);
    drain_queues(queues, "ipv4_error", m_node_ipv4_error_queues);
    drain_queues(queues, "host_nic_ingress", m_node_frame_ingress_queues);
    drain_queues(queues, "host_nic_egress", m_node_frame_egress_queues);
    drain_queues(queues, "switch_nic_ingress", m_sw_frame_ingress_queues);
    drain_queues(queues, "switch_nic_egress", m_sw_frame_egress_queues);
    // P2P channel endpoints alias NIC queues and carry scheduled arrival timestamps.
    drain_queues(queues, "channel_ingress", m_ch_frame_ingress_queues);
    drain_queues(queues, "channel_egress", m_ch_frame_egress_queues);

    connections << "node,source_ip,dst_ip,generated_bytes,generated_start_ns,"
                   "device_bytes,device_start_ns,device_end_ns,device_byte_offset,"
                   "pending_flows,acked_flows,planned_bytes,acked_bytes,"
                   "snd,una,expected_seq,record_count,has_records,unacked_packets\n";
    if (flows_map_.size() != m_nodes.size() || m_tcp_num_per_node.size() != m_nodes.size() ||
        m_tcp_connections.size() != m_nodes.size() * MAX_TCP_CONNECTION_NUM)
      throw std::runtime_error("GeDES flow and connection inventory differs from fixed fixture");
    for (size_t i = 0; i < m_nodes.size(); ++i)
    {
      // CreateFlows and BuildTCPConnections use one directed connection and one flow per node.
      const auto &generated = flows_map_.at(i);
      if (generated.size() != 1 || m_tcp_num_per_node[i] != 1)
        throw std::runtime_error("GeDES fixture changed the per-node flow or connection count");
      const auto tcp = drain_read(m_tcp_connections[i * MAX_TCP_CONNECTION_NUM]);
      const auto flow_queue = drain_read(tcp.flows);
      if (flow_queue.queue_capacity <= 1 || flow_queue.size < 0 || flow_queue.head < 0 ||
          flow_queue.size > 1 || flow_queue.head != 1 - flow_queue.size)
        throw std::runtime_error("invalid GeDES fixed-fixture flow queue state");
      const auto flow = drain_read(flow_queue.queue);
      // byte_offset is uninitialized until send_tcp_packets removes the scheduled flow.
      const auto byte_offset = flow_queue.size == 0 ? flow.byte_offset : -1;
      connections << i << ',' << tcp.src_ip << ',' << tcp.dst_ip << ','
                  << generated[0].flow_size << ',' << generated[0].timestamp << ','
                  << flow.flow_size << ',' << flow.timestamp << ',' << flow.tiimestamp_end << ','
                  << byte_offset << ',' << flow_queue.size << ',' << tcp.acked_flows << ','
                  << tcp.planned_bytes << ',' << tcp.acked_bytes << ',' << tcp.snd << ',' << tcp.una << ','
                  << tcp.expected_seq << ',' << tcp.record_num << ',' << (tcp.records != nullptr) << ','
                  << tcp.unacked_packets << '\n';
    }
    queues.close();
    connections.close();
  }

void CheckPipelineWarmup(VulkanHarness& vulkan) {
  constexpr const char* name = "PipelineWarmup";
  GraphicContext graphics;
  graphics.device = vulkan.Device();
  graphics.physical_device_properties = vulkan.RuntimeContext().GetPhysicalDeviceProperties();
  const auto path = std::filesystem::temp_directory_path() /
      ("kyty-warmup-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
  {
    PipelineCache cache(graphics, nullptr);
    vk::PipelineCache driver;
    vk::PipelineCacheCreateInfo create{};
    Require(name, "driver cache creation",
        graphics.device.createPipelineCache(&create, nullptr, &driver) == vk::Result::eSuccess,
        "could not create the regression test's driver cache");
    PipelineCacheTestAccess::SetDriverCache(cache, driver, path);
    for (u32 threads : {1u, 2u}) {
      PipelineRecord::Records records;
      auto& program = records.programs.emplace_back();
      program.stage = ShaderType::Compute;
      program.hash = 0x436f646578574152ull;
      AppendVMovU32(&program.code, 0, 1);
      AppendEnd(&program.code);
      auto& input = program.compute;
      input.threads_num[0] = input.dispatch_threads_num[0] = threads;
      input.threads_num[1] = input.threads_num[2] = 1;
      input.dispatch_threads_num[1] = input.dispatch_threads_num[2] = 1;
      input.wave_size = 32;
      input.workgroup_register = 8;
      records.compute.push_back({0, input});
      PipelineCacheTestAccess::WarmUp(cache, records);
      Require(name, "rebound stage runtime", bool(records.compute[0].compute.stage),
          "replayed pipeline lacks the program or resource snapshot pointer");
      Require(name, "pipeline created after save",
          PipelineCacheTestAccess::ComputePipelines(cache) == threads,
          "warm-up did not create each distinct workgroup pipeline");
      cache.Save();
      Require(name, "live cache survives repeated saves",
          PipelineCacheTestAccess::DriverCache(cache) == driver && std::filesystem::file_size(path) > 0,
          "saving destroyed the live driver cache or failed to write its contents");
    }
  }
  std::filesystem::remove(path);
  std::printf("[gpu]     %-32s ok\n", name);
}

#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// undef_vgprs: VGPRs no write reaches read as UndefU32 (see ShaderUndefVgprs) instead of 0.
void RewriteToSsa(const BlockList& blocks, bool undef_vgprs = false);

} // namespace Libs::Graphics::ShaderRecompiler::IR

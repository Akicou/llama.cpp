#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_quasar::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Quasar alternates Qwen3.5 Gated Delta Net blocks with FLA GLA blocks.
    // Both block kinds are recurrent, so no KV cache is needed.
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = true;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_quasar::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t gdn_head_k_dim = hparams.ssm_d_state;
    const int64_t gdn_head_v_dim = hparams.ssm_d_state;
    const int64_t gdn_n_k_heads  = hparams.ssm_n_group;
    const int64_t gdn_n_v_heads  = hparams.ssm_dt_rank;
    const int64_t gdn_key_dim    = gdn_head_k_dim * gdn_n_k_heads;
    const int64_t gdn_value_dim  = gdn_head_v_dim * gdn_n_v_heads;
    const int64_t gdn_conv_dim   = 2*gdn_key_dim + gdn_value_dim;

    const int64_t gla_head_dim   = 64;
    const int64_t gla_n_heads    = n_head;
    const int64_t gla_n_kv_heads = n_head_kv;
    const int64_t gla_key_dim    = gla_head_dim * gla_n_kv_heads;
    const int64_t gla_inner_dim  = gla_head_dim * gla_n_heads;
    const int64_t gla_gate_rank  = 16;

    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

        const bool is_gdn = ((i + 1) % full_attn_interval != 0);
        if (is_gdn) {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", i), { n_embd, 2*gdn_key_dim + gdn_value_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", i), { n_embd, gdn_value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), { hparams.ssm_d_conv, gdn_conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         i), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", i), { n_embd, gdn_n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", i), { n_embd, gdn_n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", i), { gdn_head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", i), { gdn_value_dim, n_embd }, 0);
        } else {
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, gla_inner_dim }, 0); // g_proj
            layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), { n_embd, gla_inner_dim }, 0);
            layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), { n_embd, gla_key_dim   }, 0);
            layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), { n_embd, gla_key_dim   }, 0);
            layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), { gla_inner_dim, n_embd }, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,   "weight", i), { n_embd, gla_gate_rank }, 0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,   "weight", i), { gla_gate_rank, gla_key_dim }, 0);
            layer.ssm_dt   = create_tensor(tn(LLM_TENSOR_SSM_DT,    "bias",   i), { gla_key_dim }, 0); // gk_proj.1.bias
            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM,  "weight", i), { gla_head_dim }, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_quasar::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_quasar::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_rs_inp();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (model.layers[il].ssm_f_a != nullptr) {
            cur = build_layer_attn_gla(inp, cur, il);
        } else {
            cur = build_layer_attn_linear(inp, cur, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_quasar::graph::build_qkvz(ggml_tensor * input, int il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_quasar::graph::build_norm_gated(ggml_tensor * input, ggml_tensor * weights, ggml_tensor * gate, int layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);
    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_quasar::graph::build_layer_attn_linear(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt));
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    const int64_t conv_kernel_size = model.layers[il].ssm_conv1d->ne[0];
    const int64_t conv_channels    = d_inner + 2*hparams.ssm_n_group*hparams.ssm_d_state;
    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);

    ggml_tensor * last_conv_states = ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    ggml_tensor * state_update_target = ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels,
            n_seqs, conv_states_all->nb[1], kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_output = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, model.layers[il].ssm_conv1d));

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_output->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_output, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_output, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_output));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_output, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_output->type, 2 * head_k_dim * num_k_heads));

    q_conv = ggml_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = ggml_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state,
            ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
    return cur;
}

ggml_tensor * llama_model_quasar::graph::build_layer_attn_gla(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_tokens_all = n_seq_tokens * n_seqs;
    const int64_t head_dim     = 64;
    const int64_t n_heads      = hparams.n_head();
    const int64_t n_kv_heads   = hparams.n_head_kv();
    const int64_t inner_dim    = head_dim * n_heads;
    const int64_t kv_dim       = head_dim * n_kv_heads;
    const int64_t state_size   = head_dim * inner_dim;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());

    ggml_tensor * q = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    ggml_tensor * k = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    ggml_tensor * v = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);

    ggml_tensor * g = build_lora_mm(model.layers[il].ssm_f_a, cur);
    g = build_lora_mm(model.layers[il].ssm_f_b, g);
    g = ggml_add(ctx0, g, model.layers[il].ssm_dt);
    // FLA GLA uses log-sigmoid gate logits divided by the default normalizer (16).
    // ggml_gated_linear_attn expects the recurrent multiplier directly.
    g = ggml_exp(ctx0, ggml_scale(ctx0, ggml_log(ctx0, ggml_sigmoid(ctx0, g)), 1.0f/16.0f));

    q = ggml_reshape_3d(ctx0, q, head_dim, n_heads, n_tokens_all);
    k = ggml_reshape_4d(ctx0, k, head_dim, 1, n_kv_heads, n_tokens_all);
    v = ggml_reshape_4d(ctx0, v, head_dim, 1, n_kv_heads, n_tokens_all);
    g = ggml_reshape_4d(ctx0, g, head_dim, 1, n_kv_heads, n_tokens_all);

    if (n_kv_heads != n_heads) {
        GGML_ASSERT(n_heads % n_kv_heads == 0);
        ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_heads / n_kv_heads, n_kv_heads, n_tokens_all);
        k = ggml_repeat(ctx0, k, tmp);
        v = ggml_repeat(ctx0, v, tmp);
        g = ggml_repeat(ctx0, g, tmp);
    }

    k = ggml_reshape_3d(ctx0, k, head_dim, n_heads, n_tokens_all);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_heads, n_tokens_all);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_heads, n_tokens_all);

    const auto kv_head = mctx_cur->get_head();
    ggml_tensor * states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state_full = build_rs(inp, states_all, hparams.n_embd_s(), n_seqs);
    ggml_tensor * state = ggml_view_2d(ctx0, state_full, state_size, n_seqs, state_full->nb[1], 0);

    ggml_tensor * gla = ggml_gated_linear_attn(ctx0, k, v, q, g, state, powf(float(head_dim), -0.5f));
    ggml_tensor * output = ggml_view_1d(ctx0, gla, inner_dim * n_tokens_all, 0);
    ggml_tensor * new_state = ggml_view_2d(ctx0, gla, state_size, n_seqs, state_size * ggml_element_size(gla),
            inner_dim * n_tokens_all * ggml_element_size(gla));

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state,
            ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1],
                kv_head * hparams.n_embd_s() * ggml_element_size(states_all))));

    output = ggml_reshape_3d(ctx0, output, head_dim, n_heads, n_tokens_all);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, cur, model.layers[il].wqkv_gate_s);
    z = ggml_reshape_3d(ctx0, z, head_dim, n_heads, n_tokens_all);
    output = build_norm_gated(output, model.layers[il].ssm_norm, z, il);
    output = ggml_reshape_2d(ctx0, output, inner_dim, n_tokens_all);

    cur = build_lora_mm(model.layers[il].wo, output, model.layers[il].wo_s);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens_all);
    return cur;
}

ggml_tensor * llama_model_quasar::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);
    return cur;
}

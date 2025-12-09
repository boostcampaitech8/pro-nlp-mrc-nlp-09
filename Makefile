# MRC 프로젝트 Makefile
# 실험 실행을 간편하게 관리하기 위한 유틸리티

.PHONY: help train inference train-pipeline eval-val eval-test batch check-config list-active check-active gpu-status clean-checkpoints compare-results

# 기본 설정
PYTHON := python
ACTIVE_DIR := configs/active
OUTPUT_DIR := ./outputs
USER := dahyeong
# USER 변수: 사용자 이름 (필요시 변경)

# 색상 출력
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ 도움말

help: ## 이 도움말 메시지 출력
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  MRC 프로젝트 Makefile 사용 가이드$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make train CONFIG=configs/my_exp.yaml"
	@echo "  make train-pipeline CONFIG=configs/my_exp.yaml  # train + test inference"
	@echo "  make batch                                      # configs/active/*.yaml 모두 실행"
	@echo "  make eval-val CONFIG=configs/my_exp.yaml        # validation 분석"
	@echo ""

##@ 실험 실행

train: ## 학습만 (CONFIG=path/to/config.yaml 필수)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make train CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🚀 Starting training with $(CONFIG)$(NC)"
	$(PYTHON) run.py --mode train --config $(CONFIG)

inference: ## 추론만 (CONFIG=path/to/config.yaml 필수)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make inference CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔍 Starting inference with $(CONFIG)$(NC)"
	$(PYTHON) run.py --mode inference --config $(CONFIG)

train-pipeline: ## 학습 + test inference (기본 workflow)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make train-pipeline CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔄 Starting train + test inference pipeline$(NC)"
	$(PYTHON) run.py --mode pipeline --config $(CONFIG)

eval-val: ## Validation 분석 (gold vs retrieval 비교)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make eval-val CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)📊 Evaluating validation set (gold vs retrieval)$(NC)"
	@echo "$(YELLOW)Step 1: Inference with gold context...$(NC)"
	@$(PYTHON) -c "import yaml, sys; \
		config = yaml.safe_load(open('$(CONFIG)')); \
		config['inference_split'] = 'validation'; \
		config['eval_retrieval'] = False; \
		yaml.dump(config, sys.stdout)" > /tmp/val_gold_config.yaml
	@$(PYTHON) run.py --mode inference --config /tmp/val_gold_config.yaml
	@echo ""
	@echo "$(YELLOW)Step 2: Inference with retrieval...$(NC)"
	@$(PYTHON) -c "import yaml, sys; \
		config = yaml.safe_load(open('$(CONFIG)')); \
		config['inference_split'] = 'validation'; \
		config['compare_retrieval'] = True; \
		yaml.dump(config, sys.stdout)" > /tmp/val_retrieval_config.yaml
	@$(PYTHON) run.py --mode inference --config /tmp/val_retrieval_config.yaml
	@rm -f /tmp/val_gold_config.yaml /tmp/val_retrieval_config.yaml
	@echo "$(GREEN)✅ Validation evaluation completed!$(NC)"

eval-test: ## Test inference (retrieval 필수, 이미 학습된 모델)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make eval-test CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔍 Running test inference with retrieval$(NC)"
	$(PYTHON) run.py --mode inference --config $(CONFIG)

batch: ## configs/active/*.yaml 모두 순차 실행 (train-pipeline)
	@echo "$(BLUE)🚀 Starting batch mode with all configs in $(ACTIVE_DIR)/$(NC)"
	@if [ -z "$$(ls -A $(ACTIVE_DIR)/*.yaml 2>/dev/null)" ]; then \
		echo "$(RED)❌ Error: $(ACTIVE_DIR)/ 폴더에 YAML 파일이 없습니다$(NC)"; \
		echo "$(YELLOW)💡 Tip: configs/*.yaml 파일을 $(ACTIVE_DIR)/로 복사하세요$(NC)"; \
		exit 1; \
	fi
	@total=$$(ls -1 $(ACTIVE_DIR)/*.yaml | wc -l); \
	echo "$(YELLOW)📋 Total configs: $$total$(NC)"; \
	echo ""; \
	count=0; \
	for config in $(ACTIVE_DIR)/*.yaml; do \
		count=$$((count+1)); \
		echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"; \
		echo "$(GREEN)📦 [$$count/$$total] Processing: $$config$(NC)"; \
		echo "$(GREEN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"; \
		$(MAKE) train-pipeline CONFIG=$$config || echo "$(RED)❌ Failed: $$config$(NC)"; \
		echo ""; \
	done; \
	echo "$(GREEN)✅ Batch processing completed: $$count configs$(NC)"

##@ 설정 관리

list-active: ## configs/active/ 폴더의 설정 파일 목록
	@echo "$(BLUE)📋 Active configs (will be used in 'make batch'):$(NC)"
	@if [ -z "$$(ls -A $(ACTIVE_DIR)/*.yaml 2>/dev/null)" ]; then \
		echo "  $(YELLOW)(none)$(NC)"; \
		echo ""; \
		echo "$(YELLOW)💡 Tip: configs/*.yaml 파일을 $(ACTIVE_DIR)/로 복사하세요$(NC)"; \
	else \
		ls -1 $(ACTIVE_DIR)/*.yaml | sed 's|^|  ✓ |'; \
		echo ""; \
		echo "$(GREEN)Total: $$(ls -1 $(ACTIVE_DIR)/*.yaml | wc -l) configs$(NC)"; \
	fi

check-config: ## YAML 설정 파일 유효성 검증 (CONFIG 필수)
ifndef CONFIG
	@echo "$(RED)❌ Error: CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make check-config CONFIG=configs/my_experiment.yaml$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔍 Validating $(CONFIG)...$(NC)"
	@$(PYTHON) -c "from transformers import HfArgumentParser; from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments; parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)); parser.parse_yaml_file('$(CONFIG)'); print('✅ YAML 설정이 유효합니다!')" && echo "$(GREEN)✅ YAML 설정이 유효합니다!$(NC)" || \
	(echo "$(RED)❌ YAML 설정이 잘못되었습니다$(NC)" && exit 1)

check-active: ## configs/active/ 모든 설정 파일 유효성 검증
	@echo "$(BLUE)🔍 Validating all active configs...$(NC)"
	@if [ -z "$$(ls -A $(ACTIVE_DIR)/*.yaml 2>/dev/null)" ]; then \
		echo "$(YELLOW)⚠️  No active configs found$(NC)"; \
		exit 0; \
	fi
	@failed=0; \
	for config in $(ACTIVE_DIR)/*.yaml; do \
		echo ""; \
		echo "Checking $$config..."; \
		$(PYTHON) -c "from transformers import HfArgumentParser; from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments; parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)); parser.parse_yaml_file('$$config'); print('  ✓ Valid')" && echo "  $(GREEN)✓ Valid$(NC)" || \
		(echo "  $(RED)✗ Invalid$(NC)" && failed=$$((failed+1))); \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "$(GREEN)✅ All configs are valid!$(NC)"; \
	else \
		echo "$(RED)❌ $$failed config(s) failed validation$(NC)"; \
		exit 1; \
	fi

##@ 결과 분석

compare-results: ## 실험 결과 비교 (F1/EM 점수)
	@echo "$(BLUE)📊 Comparing experiment results:$(NC)"
	@echo ""
	@for dir in $(OUTPUT_DIR)/$(USER)/*/; do \
		if [ -f "$$dir/eval_results.json" ]; then \
			exp_name=$$(basename $$dir); \
			f1=$$($(PYTHON) -c "import json; print(json.load(open('$$dir/eval_results.json')).get('eval_f1', 'N/A'))" 2>/dev/null || echo "N/A"); \
			em=$$($(PYTHON) -c "import json; print(json.load(open('$$dir/eval_results.json')).get('eval_exact_match', 'N/A'))" 2>/dev/null || echo "N/A"); \
			printf "  %-50s F1: %-8s EM: %-8s\n" "$$exp_name" "$$f1" "$$em"; \
		fi; \
	done
	@echo ""

show-best: ## 가장 높은 EM 점수 기준 Top 5 실험 출력
	@echo "$(BLUE)🏆 Top 5 experiments (by EM score):$(NC)"
	@echo ""
	@for dir in $(OUTPUT_DIR)/$(USER)/*/; do \
		if [ -f "$$dir/eval_results.json" ]; then \
			exp_name=$$(basename $$dir); \
			f1=$$($(PYTHON) -c "import json; print(json.load(open('$$dir/eval_results.json')).get('eval_f1', 0))" 2>/dev/null || echo "0"); \
			em=$$($(PYTHON) -c "import json; print(json.load(open('$$dir/eval_results.json')).get('eval_exact_match', 0))" 2>/dev/null || echo "0"); \
			printf "%s|%s|%s\n" "$$em" "$$f1" "$$exp_name"; \
		fi \
	done | sort -t'|' -k1 -nr | head -5 | awk -F'|' '{printf "  $(GREEN)%-50s$(NC) EM: %-8s F1: %-8s\n", $$3, $$1, $$2}'
	@echo ""

compare-retrieval: ## Retrieval 성능 비교 결과 출력 (EXP 필수)
	@if [ -z "$(EXP)" ]; then \
		echo "$(RED)❌ Error: EXP is required$(NC)"; \
		echo "Usage: make compare-retrieval EXP=<experiment_name>"; \
		echo "Example: make compare-retrieval EXP=oceann315_roberta-large-korquad-v1"; \
		exit 1; \
	fi
	@exp_dir=$(OUTPUT_DIR)/$(USER)/$(EXP); \
	if [ ! -d "$$exp_dir" ]; then \
		echo "$(RED)❌ Experiment directory not found: $$exp_dir$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📊 Comparing retrieval performance for: $(EXP)$(NC)"
	@$(PYTHON) scripts/compare_retrieval.py "$$exp_dir"

##@ 유틸리티

gpu-status: ## GPU 사용 현황 확인
	@echo "$(BLUE)🖥️  GPU Status:$(NC)"
	@nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
		awk -F, '{printf "  GPU %s: %s | Util: %s%% | Mem: %s/%s MB\n", $$1, $$2, $$3, $$4, $$5}'

clean-checkpoints: ## checkpoint 폴더만 정리 (best_checkpoint_path.txt 보존)
	@echo "$(YELLOW)🧹 Cleaning checkpoint folders in $(OUTPUT_DIR)/$(USER)/...$(NC)"
	@find $(OUTPUT_DIR)/$(USER) -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Checkpoints cleaned (best checkpoint files preserved)$(NC)"

install: ## 필요한 패키지 설치
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Installation complete!$(NC)"

##@ WandB Sweep (하이퍼파라미터 튜닝)

sweep-create: ## WandB Sweep 생성 (SWEEP_CONFIG, PROJECT 필수)
ifndef SWEEP_CONFIG
	@echo "$(RED)❌ Error: SWEEP_CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make sweep-create SWEEP_CONFIG=configs/sweep_config.yaml PROJECT=mrc-sweep$(NC)"
	@exit 1
endif
ifndef PROJECT
	@echo "$(RED)❌ Error: PROJECT 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make sweep-create SWEEP_CONFIG=configs/sweep_config.yaml PROJECT=mrc-sweep$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔧 Creating WandB Sweep...$(NC)"
	$(PYTHON) run_sweep.py --create --config $(SWEEP_CONFIG) --project $(PROJECT)

sweep-run: ## WandB Sweep Agent 실행 (SWEEP_ID 필수, COUNT 선택)
ifndef SWEEP_ID
	@echo "$(RED)❌ Error: SWEEP_ID 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make sweep-run SWEEP_ID=abc123 COUNT=10$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🚀 Starting WandB Sweep Agent...$(NC)"
ifdef COUNT
	$(PYTHON) run_sweep.py --sweep_id $(SWEEP_ID) --count $(COUNT)
else
	$(PYTHON) run_sweep.py --sweep_id $(SWEEP_ID)
endif

sweep-quick: ## Sweep 생성 + 즉시 실행 (SWEEP_CONFIG, PROJECT, COUNT 필수)
ifndef SWEEP_CONFIG
	@echo "$(RED)❌ Error: SWEEP_CONFIG 변수가 필요합니다$(NC)"
	@echo "$(YELLOW)Usage: make sweep-quick SWEEP_CONFIG=configs/sweep_config.yaml PROJECT=mrc-sweep COUNT=10$(NC)"
	@exit 1
endif
ifndef PROJECT
	@echo "$(RED)❌ Error: PROJECT 변수가 필요합니다$(NC)"
	@exit 1
endif
ifndef COUNT
	@echo "$(RED)❌ Error: COUNT 변수가 필요합니다$(NC)"
	@exit 1
endif
	@echo "$(BLUE)🔧 Creating and running WandB Sweep...$(NC)"
	$(PYTHON) run_sweep.py --create --run --config $(SWEEP_CONFIG) --project $(PROJECT) --count $(COUNT)

wandb-login: ## WandB 로그인
	@echo "$(BLUE)🔐 Logging in to WandB...$(NC)"
	wandb login

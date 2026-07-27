.PHONY: \
	setup \
	uv_sync \
	create_exp \
	mark_fail \
	resume_exp \
	rename_exp \
	log_clean \
	failed_log_clean \
	clean_failed \
	preflight \
	jupyter

# =========================================================
# 0. Initial setup
# =========================================================

setup:
	@bash tools/first_setup.sh

# =========================================================
# 1. Sync Python environment
# =========================================================

uv_sync:
	@if [ -z "$(p)" ]; then \
		echo "❌ Error: partition is required."; \
		echo "Usage: make uv_sync p=<partition>"; \
		exit 1; \
	fi

	sbatch --partition=$(p) tools/uv_sync.sh

# =========================================================
# 2. Create Experiment
# =========================================================

create_exp:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: name is required."; \
		echo "Usage: make create_exp name=<exp_name>"; \
		exit 1; \
	fi

	@bash tools/create_exp.sh "$(name)"

# =========================================================
# 3. Mark Experiment as FAILED
# =========================================================

mark_fail:
	@if [ -z "$(name)" ] || [ -z "$(reason)" ]; then \
		echo "❌ Error: name and reason are required."; \
		echo "Usage:"; \
		echo "  make mark_fail name=<exp_name_or_index> reason=<reason>"; \
		exit 1; \
	fi

	@bash tools/mark_failed.sh "$(name)" "$(reason)"

# =========================================================
# 4. Resume Experiment
# =========================================================

resume_exp:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: name is required."; \
		echo "Usage:"; \
		echo "  make resume_exp name=<exp_name_or_index>"; \
		echo "  make resume_exp name=<exp_name_or_index> suffix=<suffix>"; \
		exit 1; \
	fi

	@bash tools/resume_exp.sh "$(name)" "$(suffix)"

# =========================================================
# 5. Rename Experiment
# =========================================================

rename_exp:
	@if [ -z "$(name)" ] || [ -z "$(new)" ]; then \
		echo "❌ Error: name and new are required."; \
		echo "Usage:"; \
		echo "  make rename_exp name=<exp_name_or_index> new=<new_name>"; \
		exit 1; \
	fi

	@bash tools/rename_exp.sh "$(name)" "$(new)"

# =========================================================
# 6. Clean up log files into job subdirectories
#    {job_id}_{exp_name}.out        → logs/{exp_name}/{job_id}/slurm.out
#    watcher_{job_id}.log           → logs/{exp_name}/{job_id}/watcher.log
#    （watcher_*.log は旧watch_job.sh方式の名残。新規には生成されないが、
#      既存ログの後方互換整理のためにロジックは残す）
# =========================================================

log_clean:
	@root=$$(git rev-parse --show-toplevel); \
	find "$$root/logs" -maxdepth 2 -name "*_*.out" | while read f; do \
		base=$$(basename "$$f"); \
		job_id=$$(echo "$$base" | cut -d_ -f1); \
		exp_dir=$$(dirname "$$f"); \
		dest="$$exp_dir/$$job_id"; \
		mkdir -p "$$dest"; \
		mv "$$f" "$$dest/slurm.out" && echo "  $$base → $$job_id/slurm.out"; \
	done; \
	find "$$root/logs" -maxdepth 2 -name "watcher_*.log" | while read f; do \
		base=$$(basename "$$f"); \
		job_id=$$(echo "$$base" | sed 's/watcher_//; s/\.log//'); \
		exp_dir=$$(dirname "$$f"); \
		dest="$$exp_dir/$$job_id"; \
		mkdir -p "$$dest"; \
		mv "$$f" "$$dest/watcher.log" && echo "  $$base → $$job_id/watcher.log"; \
	done

# =========================================================
# 6b. Clean up failed experiment logs and job directories
#     Deletes log folders for failed, OOM, timeout, or node fail runs,
#     and logs for experiments that have been renamed to *_FAILED_*
# =========================================================

failed_log_clean:
	@root=$$(git rev-parse --show-toplevel); \
	echo "🔍 Scanning for failed/timeout/OOM job logs..."; \
	find "$$root/logs" -name "run_metadata.yaml" | while read meta; do \
		status=$$(grep -E '^status:' "$$meta" | awk '{print $$2}' | tr -d '"' | tr -d "'"); \
		if [ "$$status" = "FAILED" ] || [ "$$status" = "TIMEOUT" ] || [ "$$status" = "OUT_OF_MEMORY" ] || [ "$$status" = "NODE_FAIL" ] || [ "$$status" = "CANCELLED" ]; then \
			job_dir=$$(dirname "$$meta"); \
			echo "  ❌ Deleting failed job log directory: $$(basename "$$job_dir") (Status: $$status)"; \
			rm -rf "$$job_dir"; \
		elif [ "$$status" = "RUNNING" ]; then \
			if [ -n "$$(find "$$meta" -mmin +1440 2>/dev/null)" ]; then \
				job_dir=$$(dirname "$$meta"); \
				echo "  ❌ Deleting hung RUNNING job log directory (inactive >24h): $$(basename "$$job_dir")"; \
				rm -rf "$$job_dir"; \
			fi; \
		fi; \
	done; \
	echo "🔍 Scanning for orphaned/failed experiment logs..."; \
	find "$$root/logs" -mindepth 1 -maxdepth 1 -type d | while read log_dir; do \
		exp_name=$$(basename "$$log_dir"); \
		if [ ! -d "$$root/experiments/$$exp_name" ] && [ "$$exp_name" != "latest" ]; then \
			echo "  ❌ Deleting failed/deleted experiment log directory: $$exp_name"; \
			rm -rf "$$log_dir"; \
		fi; \
	done; \
	echo "✅ Clean up completed."

clean_failed: failed_log_clean


# =========================================================
# 7. Start Jupyter
# =========================================================

jupyter:
	@if [ -z "$(p)" ] || [ -z "$(mem)" ]; then \
		echo "❌ Error: partition and memory are required."; \
		echo "Usage:"; \
		echo "  make jupyter p=<partition> mem=<memory>"; \
		echo "Example:"; \
		echo "  make jupyter p=a6000 mem=32g"; \
		exit 1; \
	fi

	sbatch \
		--partition=$(p) \
		--mem=$(mem) \
		tools/start_jupyter.sh

# =========================================================
# 8. Preflight validation checks before runx
# =========================================================

preflight:
	@python3 -m scripts.preflight_check
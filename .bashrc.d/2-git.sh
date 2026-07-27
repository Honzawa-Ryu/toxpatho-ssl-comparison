#!/usr/bin/env bash
# Pattern2 (daily/YYMMDD 方式) 用の git ヘルパー。
# gstart/gpush をシェル関数として提供し、_setup_git_config を
# 「mizuno-group への直接コミット禁止」のみに絞ったもの。
#
# これらはユーザーがホスト側で実行する関数です。agentは実行しません。
#
# 使い方: この内容を ~/.bashrc など、テンプレート外のあなた個人の設定に
# source してください（テンプレート/capsuleリポジトリ本体には含めない）。

# ==========================================
# Git 初回セットアップ（最小限）
# ==========================================

_setup_git_config() {
    local hooks_dir="${OHTO_DIR:-$HOME}/.git-hooks"

    if [ -f "${hooks_dir}/pre-commit" ]; then
        return 0
    fi

    mkdir -p "$hooks_dir"

    cat > "${hooks_dir}/pre-commit" << 'HOOK_EOF'
#!/bin/bash
REMOTE_URL=$(git config --get remote.origin.url)
OWNER=$(echo "$REMOTE_URL" | sed -E 's|.*[:/]([^/]+)/[^/]+/?(.git)?$|\1|')

if [[ "$OWNER" == "mizuno-group" ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌  エラー: コミットがブロックされました"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "このリポジトリはmizuno-groupであり、コミットは禁止されています。"
  echo "リモートURL: $REMOTE_URL"
  echo ""
  echo "ブロックを解除: git commit --no-verify"
  echo ""
  exit 1
fi

exit 0
HOOK_EOF

    chmod +x "${hooks_dir}/pre-commit"
    git config --global core.hooksPath "$hooks_dir"

    echo "✅ git pre-commit hook を設定しました（mizuno-group への直接コミットを禁止）"
    echo "   core.hooksPath: $(git config --global core.hooksPath)"
}

# 対話シェルでのみ初回セットアップを走らせる
# (_is_interactive_shell は .bashrc.d/0-bootstrap.sh 由来。テンプレート外に
#  持ち出す場合はこの関数も一緒に持ち出すか、下の1行に置き換える)
# _is_interactive_shell() { [[ $- == *i* ]]; }
if _is_interactive_shell 2>/dev/null; then
    _setup_git_config
fi

# ==========================================
# gstart: 1日の作業開始時にユーザーが実行する（agentは実行しない）
#   1. main を最新化する
#   2. 今日の daily/YYMMDD ブランチを作成（既にあれば checkout のみ）
# ==========================================

gstart() {
    local repo_root
    repo_root="$(git rev-parse --show-toplevel)" || return 1
    cd "$repo_root" || return 1

    if [ -n "$(git status --porcelain)" ]; then
        echo "エラー: 未コミットの変更があります。commit/stash してから gstart を実行してください。" >&2
        return 1
    fi

    git checkout main || return 1
    git pull origin main || return 1

    local today branch
    today="$(date +%y%m%d)"
    branch="daily/${today}"

    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        echo "ブランチ ${branch} は既に存在します。checkout します。"
        git checkout "${branch}" || return 1
        # 他のマシン/runで先に daily/YYMMDD が作られていた場合に備えて、
        # remote に同名ブランチがあれば取り込む（ffのみ、conflictがあれば手動対応）
        if git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
            git pull --ff-only origin "${branch}" || {
                echo "警告: origin/${branch} との fast-forward pull に失敗しました。手動で確認してください。" >&2
            }
        fi
    else
        git checkout -b "${branch}" || return 1
        echo "新しいブランチ ${branch} を作成しました。"
    fi

    echo
    echo "現在のブランチ: $(git rev-parse --abbrev-ref HEAD)"
    echo "この状態で agentrun を起動してください（例: agentrun assist --repo ${repo_root} ...）。"
}

# ==========================================
# gpush: 1日の作業終了時にユーザーが実行する（agentは実行しない）
#   デフォルト:
#     1. 現在の daily/YYMMDD ブランチを origin へ push
#     2. main を最新化して daily/YYMMDD を --no-ff でマージ
#     3. main を origin へ push
#   `gpush -gh` : 2/3の代わりに GitHub PR を作成し、確認後にマージする
#     （gh CLI が必要。https://cli.github.com/ でインストール・`gh auth login`
#      しておくこと）
#   実行前提: 現在のブランチが daily/YYMMDD であること
# ==========================================

gpush() {
    local repo_root branch use_gh=0
    if [[ "${1:-}" == "-gh" || "${1:-}" == "--gh" ]]; then
        use_gh=1
    fi

    repo_root="$(git rev-parse --show-toplevel)" || return 1
    cd "$repo_root" || return 1

    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$branch" != daily/* ]]; then
        echo "エラー: 現在のブランチ (${branch}) は daily/* ではありません。gpush は daily ブランチ上で実行してください。" >&2
        return 1
    fi

    if [ -n "$(git status --porcelain)" ]; then
        echo "エラー: 未コミットの変更があります。commit してから gpush を実行してください。" >&2
        return 1
    fi

    echo "1. ${branch} を origin へ push します。"
    git push origin "${branch}" || return 1

    if [ "$use_gh" -eq 1 ]; then
        if ! command -v gh >/dev/null 2>&1; then
            echo "エラー: gh コマンドが見つかりません。https://cli.github.com/ からインストールし、gh auth login してください。" >&2
            return 1
        fi

        local remote_url owner repo
        remote_url=$(git config --get remote.origin.url)
        owner=$(echo "$remote_url" | sed -E 's#.*[:/]([^/]+)/[^/]+/?(\.git)?$#\1#')
        repo=$(echo "$remote_url"  | sed -E 's#.*/([^/.]+)(\.git)?$#\1#')

        echo "2. PR を作成します: ${owner}/${repo}  base=main  head=${branch}"
        local gpr_err
        gpr_err="$(mktemp)"
        if ! gh pr create --repo "${owner}/${repo}" --base main --head "${branch}" --fill 2>"$gpr_err"; then
            if grep -q "already exists" "$gpr_err"; then
                echo "ℹ️ 既存の PR を使用します"
            else
                cat "$gpr_err" >&2
                rm -f "$gpr_err"
                return 1
            fi
        fi
        rm -f "$gpr_err"

        local pr_url
        pr_url=$(gh pr view "${branch}" --repo "${owner}/${repo}" --json url -q .url)
        echo "🔗 ${pr_url}"
        echo "この PR を merge しますか？ (y/n)"
        read -r ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            gh pr merge "${branch}" --repo "${owner}/${repo}" --merge --delete-branch=false || return 1
            echo "✅ main にマージしました。"
        else
            echo "💡 あとで手動マージ: ${pr_url}"
        fi

        echo
        echo "次回は gstart で新しい daily/YYMMDD ブランチを作成してください。"
        return 0
    fi

    echo "2. main を最新化してマージします。"
    git checkout main || return 1
    git pull origin main || return 1
    git merge --no-ff "${branch}" -m "merge ${branch} into main" || return 1

    echo "3. main を origin へ push します。"
    git push origin main || return 1

    echo
    echo "完了しました。main には ${branch} の変更が統合されています。"
    echo "次回は gstart で新しい daily/YYMMDD ブランチを作成してください。"
}
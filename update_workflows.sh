for file in .github/workflows/opencode-easy.yml .github/workflows/opencode-hard.yml .github/workflows/opencode-review.yml; do
  sed -i '/- name: Run OpenCode/i \      - name: Configure Git\n        run: |\n          git config --global user.name "github-actions[bot]"\n          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"\n' "$file"
done

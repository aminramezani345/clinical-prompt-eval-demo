# Push to your GitHub

These steps put the repo at `https://github.com/aminramezani345/clinical-prompt-eval-demo`.
You only need to do this once.

## 1. Create the empty repo on GitHub

Go to https://github.com/new and create a repo named **clinical-prompt-eval-demo**.
- Owner: aminramezani345
- Visibility: your choice. Public is fine, there is no PHI in this repo.
- Do NOT initialise with a README, .gitignore, or license. You want the repo
  to start empty so the push below is the first commit.

## 2. Initialise and push from your machine

Open a terminal and run, from the folder this README lives in:

```bash
cd "<path to>/clinical-prompt-eval-demo"
git init
git checkout -b main
git add .
git commit -m "Initial commit: prompt eval CI demo"
git remote add origin git@github.com:aminramezani345/clinical-prompt-eval-demo.git
git push -u origin main
```

If you use HTTPS instead of SSH, swap the remote line for:

```bash
git remote add origin https://github.com/aminramezani345/clinical-prompt-eval-demo.git
```

## 3. Watch the CI

Open the **Actions** tab on the new repo. The first push will run
`prompt-eval.yml` automatically because it is also triggered by
`workflow_dispatch`, or you can trigger it from the Actions UI.

## 4. Try a real PR to see the gate in action

Make a branch that "regresses" v2 by reintroducing the hallucinated
anticoagulant for case c005, open a PR against main, and you should see the
hard gate `is_hallucination` block the merge with a PR comment.

```bash
git checkout -b regress-c005
# Edit evals/model.py, change the v2 c005 entry to mention warfarin instead
# of apixaban. Commit and push.
git commit -am "Demo: regress v2 c005 to show the gate working"
git push -u origin regress-c005
```

Then open a PR on GitHub. The Actions run will fail and post a comment
showing exactly which metric tripped and by how much.

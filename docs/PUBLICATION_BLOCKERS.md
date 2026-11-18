# Publication status — 2026-11-18

The local feature branch is `feature/mobile-skinxai-final-integration`, based on remote main `55b155650f94e53c8c127f54e006031f88770d99`. Original development files and research archives remain at their original locations; the publication copy is an isolated Git worktree.

## Remote publication is blocked

The Git CLI has no noninteractive GitHub credentials. The existing authenticated GitHub connector can read the repository but both `git/blobs` and `git/refs` creation return HTTP403: `Resource not accessible by integration`. No blob, remote branch, PR, merge, tag or release was created. A repository connection with Contents write permission (and workflow permission for the supplied workflow), or an authenticated Git CLI, is required. No password/token has been requested, printed or stored.

This is not a request for renewed merge authorization. The user already authorized a normal squash merge after all safety/quality checks pass. That authorization does not bypass GitHub permissions or the remaining build checks.

## Remaining gates

1. Push the reviewed local branch after secure authentication is available, then open a PR against main.
2. Execute all supplied CI jobs, including real Linux Flutter interpreter tests and unsigned iOS/native-XCFramework build. These jobs have NOT executed in GitHub during this task. If a job fails, repair and rerun it before merge; never bypass required checks or branch protections.
3. Recheck current main, branch protections, required status checks and the exact PR head before normal squash merge. No force push is needed or authorized.
4. The main-branch workflow may publish the prerelease only after all jobs pass. Check the archive and model hash on download. Source code ZIP requires building the native iOS framework; a verified complete release ZIP additionally includes it.
5. Physical Android/iPhone inference, camera/Photos byte behavior, AOT execution and device latency remain unmeasured. HEIC/HEIF and on-device Grad-CAM remain unsupported. An unsigned CI build cannot establish those outcomes.

The local archive is an extractable research/deployment **candidate**, not a claim of a published or fully device-tested mobile release. It includes native iOS source/build instructions, not an unverified prebuilt XCFramework. No failed Float16 model is shipped.

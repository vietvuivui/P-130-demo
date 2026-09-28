# Đóng góp cho AI20K Agent Template

## Quy ước nội bộ nhóm P-130

Áp dụng cho 4 thành viên khi làm trên repo của đội (phần bên dưới là hướng dẫn chung của template BTC).

1. **Nhánh**: mỗi việc một nhánh tách từ nhánh đang dùng chung, đặt tên `feat/…`, `fix/…`, `docs/…`
   (ví dụ `feat/label-propagation`). Không push thẳng lên `main`.
2. **Commit** theo Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
3. **Trước khi push**: `ruff check src tests` và `pytest` phải sạch.
4. **Mỗi lần push thêm một mục vào đầu [`CHANGELOG.md`](CHANGELOG.md)**: làm gì, file chính, ảnh hưởng tới người
   khác, cách kiểm tra, việc còn dở. Đổi schema / config / API thì bắt buộc ghi ở "Ảnh hưởng tới người khác".
5. **Cuối ngày** thêm một dòng vào [`WORKLOG.md`](WORKLOG.md) (deliverable #9 của BTC).

Cảm ơn bạn đã quan tâm. Repo này gồm hai phần với quy trình review khác nhau:

| Phần | Đường dẫn | Ai duyệt |
|------|-----------|----------|
| Code template | `src/`, `tests/`, `scripts/`, Dockerfile… | Maintainer bất kỳ |
| **Technical Book** | `docs/guide/**` | **Bắt buộc** @AI20K-Build-Phase/book-maintainers |

## Technical Book — lưu ý quan trọng

Thư mục `docs/guide/` là **nguồn** của Technical Book đăng tại
[phoenix.note.transformerlabs.ai/technical-book](https://phoenix.note.transformerlabs.ai/technical-book).
Nội dung được merge vào `main` sẽ đồng bộ lên trang đó.

Vì vậy mọi PR đụng vào `docs/guide/**` đều cần review của
[@AI20K-Build-Phase/book-maintainers](https://github.com/orgs/AI20K-Build-Phase/teams/book-maintainers)
— xem [`.github/CODEOWNERS`](.github/CODEOWNERS). GitHub tự gán reviewer, bạn
không cần làm gì thêm.

Khi sửa nội dung sách, mô tả rõ trong PR: **chương nào**, **sửa gì**, và **vì sao**.

## Quy trình

1. Fork repo, tạo nhánh từ `main`: `git checkout -b docs/sua-chuong-04`
2. Commit theo [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`
3. Chạy kiểm tra trước khi push:
   ```bash
   ruff check .
   pytest
   ```
4. Mở PR vào `main`. CI (`.github/workflows/ci.yml`) phải xanh.

## Không commit thông tin bí mật

Repo này là **public**. Tuyệt đối không commit API key, token, hay credential —
kể cả trong `.env.example`, comment, hay file docs.

- `.env.example` chỉ chứa **placeholder** (ví dụ `your-api-key-here`).
- Giá trị thật đặt trong `.env` — file này đã nằm trong `.gitignore`.
- Nếu lỡ commit, đọc [SECURITY.md](SECURITY.md): key coi như đã lộ và **phải
  được xoay (rotate)**, xoá commit thôi là chưa đủ.

## Báo lỗi

Mở [issue](https://github.com/AI20K-Build-Phase/starter-code-template/issues) kèm:
môi trường (OS, Python version), các bước tái hiện, log lỗi.

Lỗ hổng bảo mật thì **không** mở issue công khai — làm theo [SECURITY.md](SECURITY.md).

## Quy tắc ứng xử

Tham gia repo này đồng nghĩa với việc bạn đồng ý tuân thủ
[Quy tắc ứng xử](CODE_OF_CONDUCT.md).

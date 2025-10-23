
# 使用网易 163 邮箱的最简部署

本项目每天 08:00（America/Los_Angeles）发一封“线索链接 + 主题标签 + 建议角度”的邮件，发件箱为 **163 邮箱 SMTP**。

## 一次性三步（GitHub Actions）
1. **开启 163 邮箱的 IMAP/SMTP，并获取“客户端授权密码”**  
   - 登录 163 网页端 → 设置 → POP3/IMAP/SMTP → 开启服务 → 生成 **授权密码**（不是登录密码）。
2. **建 GitHub 仓库并上传本项目文件**。
3. **在仓库 Settings → Secrets → Actions 新建 Secret**：  
   - 名称：`SMTP_PASSWORD`，值：你在上一步生成的 **授权密码**。

> 收件人与发件人：编辑 `config.yaml`，把 `from_email/user` 改为你的 163 邮箱，把 `to_emails` 改为你的收件人列表（多个用逗号分隔）。

## 工作流何时运行？
- GitHub 使用 UTC，工作流在 **15:00 / 16:00 UTC** 运行；脚本内部会检查洛杉矶当地是否 **08:00**，不是就直接退出（避免夏令时出错）。

## 本地运行（可选）
```bash
pip install -r requirements.txt
export SMTP_PASSWORD="你的163授权密码"
python daily_infoq_leads.py
```

## 修改/扩展
- 权重、话题篮子、白/黑名单：`config.yaml`
- 关键词与建议角度：`keywords.yaml`

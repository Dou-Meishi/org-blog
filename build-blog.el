(require 'org-static-blog)

(setq dms/org-static-blog-root-dir "/home/dou/Documents/2024-01-24-MyOrgBlog/")

(setq org-static-blog-publish-title "Dou Meishi's Blog")
(setq org-static-blog-publish-url "https://dou-meishi.github.io/org-blog/")
(setq org-static-blog-publish-directory (format "%s" dms/org-static-blog-root-dir))
(setq org-static-blog-posts-directory (format "%s" dms/org-static-blog-root-dir))
(setq org-static-blog-drafts-directory (format "%sdrafts" dms/org-static-blog-root-dir))
(setq org-static-blog-page-header (with-temp-buffer
  (insert-file-contents (format "%sstatic/header.html" dms/org-static-blog-root-dir))
  (buffer-string)))
(setq org-static-blog-page-preamble (with-temp-buffer
  (insert-file-contents (format "%sstatic/preamble.html" dms/org-static-blog-root-dir))
  (buffer-string)))
(setq org-static-blog-page-postamble (with-temp-buffer
  (insert-file-contents (format "%sstatic/postamble.html" dms/org-static-blog-root-dir))
  (buffer-string)))
(setq org-static-blog-index-front-matter
      "<h1 class=title> Recent Posts </h1>")
(setq org-static-blog-enable-tags t)
(setq org-static-blog-use-preview t)
(setq org-static-blog-preview-ellipsis "...")

;; publish
(org-static-blog-publish t)

(message "Publish complete!")

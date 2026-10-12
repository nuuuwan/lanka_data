import { ListItemIcon, MenuItem } from "@mui/material";

export default function AppExternalLinkMenuItem({
  href,
  onClose,
  icon: Icon,
  children,
}) {
  return (
    <MenuItem
      component="a"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={onClose}
    >
      <ListItemIcon>
        <Icon fontSize="small" />
      </ListItemIcon>
      {children}
    </MenuItem>
  );
}

use crate::error::{Error, Result};
use crate::ffi;
use libc::c_char;
use std::ffi::{CStr, CString};

pub(crate) fn to_cstring(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::NulByte(value.to_string()))
}

pub(crate) unsafe fn take_owned_c_string(ptr: *mut c_char) -> Result<String> {
    if ptr.is_null() {
        return Err(Error::NullPointer("C string result"));
    }

    let text = unsafe {
        CStr::from_ptr(ptr)
            .to_str()
            .map_err(|_| Error::Utf8("C string result".to_string()))?
            .to_owned()
    };
    unsafe {
        ffi::libia_string_free(ptr);
    }
    Ok(text)
}

pub(crate) unsafe fn read_c_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_owned()) }
}

pub(crate) fn normalize_prop_name(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() {
        return String::new();
    }

    let mut out = if name.starts_with("--") {
        name.to_string()
    } else if name.starts_with('-') {
        format!("-{}", name)
    } else {
        format!("--{}", name)
    };

    out = out.replace('_', "-");
    out
}

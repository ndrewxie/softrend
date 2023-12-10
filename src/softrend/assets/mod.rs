use lazy_static::lazy_static;
use std::collections::HashMap;

pub const TEX_SIZE: usize = 128;
#[derive(Clone, Debug)]
pub struct Texture(pub Vec<u8>);

lazy_static! {
    pub static ref TEXTURES: HashMap<&'static str, Texture> = {
        let mut m = HashMap::new();
        m.insert("joemama", Texture(include_bytes!("./joemama.rgba").to_vec()));
        m
    };
}
